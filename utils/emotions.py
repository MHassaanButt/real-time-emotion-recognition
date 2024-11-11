
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
from tqdm import tqdm
from moviepy.editor import VideoFileClip, ImageSequenceClip
from facenet_pytorch import MTCNN
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    AutoConfig,
)
from PIL import Image, ImageDraw
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def set_cache_directory():
    os.environ["XDG_CACHE_HOME"] = os.getcwd()
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd()

def set_device():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Running on device: {}".format(device))
    return device

def load_video(scene):
    clip = VideoFileClip(scene)
    vid_fps = clip.fps
    video = clip.without_audio()
    video_data = np.array(list(video.iter_frames()))
    return clip, vid_fps, video, video_data

def download_weights(model_name):
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    # Download model weights to the current working directory
    extractor.save_pretrained(os.getcwd())
    model.save_pretrained(os.getcwd())

def detect_emotions(image, mtcnn, extractor, model, emotions):
    temporary = image.copy()
    sample = mtcnn.detect(temporary)

    if sample[0] is not None:
        box = sample[0][0]
        face = temporary.crop(box)

        inputs = extractor(images=face, return_tensors="pt")
        outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        id2label = AutoConfig.from_pretrained("models").id2label
        probabilities = probabilities.detach().numpy().tolist()[0]
        class_probabilities = {
            id2label[i]: prob for i, prob in enumerate(probabilities)
        }
        return face, class_probabilities

    return None, None


def detect_emotions_v2(image, mtcnn, extractor, model):
    temporary = image.copy()
    sample = mtcnn.detect(temporary)

    if sample[0] is not None:
        box = sample[0][0]
        face = temporary.crop(box)

        inputs = extractor(images=face, return_tensors="pt")
        outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        id2label = AutoConfig.from_pretrained("models").id2label
        probabilities = probabilities.detach().numpy().tolist()[0]
        class_probabilities = {
            id2label[i]: prob for i, prob in enumerate(probabilities)
        }
        return face, class_probabilities

    return None, None


def create_combined_image(face, class_probabilities):
    colors = {
        "angry": "red",
        "unconscious": "green",
        "painful": "gray",
        "happy": "yellow",
        "neutral": "purple",
        "sad": "blue",
        "surprise": "orange",
    }
    
    key_replacement = {'disgust': 'unconscious', 'fear': 'painful'}
    for old_key, new_key in key_replacement.items():
        if old_key in class_probabilities:
            class_probabilities[new_key] = class_probabilities.pop(old_key)        
   
    palette = [colors[label] for label in class_probabilities.keys()]

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].imshow(np.array(face))
    axs[0].axis("off")

    sns.barplot(
        ax=axs[1],
        y=list(class_probabilities.keys()),
        x=[prob * 100 for prob in class_probabilities.values()],
        palette=palette,
        orient="h",
    )
    axs[1].set_xlabel("Probability (%)")
    axs[1].set_title("Emotion Probabilities")
    axs[1].set_xlim([0, 100])

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return img

def key_parser(all_class_probabilities, key_replacement={'disgust': 'unconscious', 'fear': 'painful'}):
    # Replace keys based on the mapping
    for item in all_class_probabilities:
        for old_key, new_key in key_replacement.items():
            if old_key in item:
                item[new_key] = item.pop(old_key)
    return all_class_probabilities

def process_video(input_video, output_option="graph"):
    try:
        set_cache_directory()
        device = set_device()

        # Load your video
        clip, vid_fps, video, video_data = load_video(input_video)
        print("Video loaded")
        # Download weights to the current working directory
        try:
            pass
            # download_weights("models")
        except Exception as e:
            print(str(e))
        print("weights downloaded")
        # Initialize MTCNN model for single face cropping
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=200,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=False,
            device=device,
        )
        absolute_path = "/home/pb/hb/real-time-emotion-recognition/utils/models"
        print("os.getcwd()",os.getcwd())
        print(os.listdir(os.getcwd()))
        try:
            print(os.path.exists(absolute_path),os.listdir(absolute_path))
        except Exception as e:
            print(str(e))
        print("MTCNN loaded")
        # Load the pre-trained model and feature extractor
        # extractor = AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression")
        # try:
        extractor = AutoFeatureExtractor.from_pretrained(absolute_path)
        
        # model = AutoModelForImageClassification.from_pretrained(
        #     "trpakov/vit-face-expression"
        # )
        model = AutoModelForImageClassification.from_pretrained(
            absolute_path
        )
        print("Model loaded")
        # except Exception as e:
        #     print(str(e))
        print("Extractor and model loaded",extractor, model)
        # Define a list of emotions
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

        # List to hold the combined images
        combined_images = []

        # Create a list to hold the class probabilities for all frames
        all_class_probabilities = []
        print("Checkpoint 1")
        # Loop over video frames
        for i, frame in tqdm(
            enumerate(video_data), total=len(video_data), desc="Processing frames"
        ):
            try:
                # Convert frame to uint8
                frame = frame.astype(np.uint8)

                # Call detect_emotions to get face and class probabilities
                if i % 2 == 0:
                    face, class_probabilities = detect_emotions(
                        Image.fromarray(frame), mtcnn, extractor, model, emotions
                    )
                else:
                    face, class_probabilities = detect_emotions_v2(
                        Image.fromarray(frame), mtcnn, extractor, model
                    )

                # If a face was found
                if face is not None:
                    # Create combined image for this frame
                    combined_image = create_combined_image(face, class_probabilities)

                    # Append combined image to the list
                    combined_images.append(combined_image)
                else:
                    # If no face was found, set class probabilities to None
                    class_probabilities = {emotion: None for emotion in emotions}
            except Exception as e:
                print(str(e))
            # Append class probabilities to the list
            all_class_probabilities.append(class_probabilities)
        all_class_probabilities=key_parser(all_class_probabilities)    
        print("All class probabilities",all_class_probabilities)
        # Output based on the selected option
        if output_option == "graph":
            # Display line plot
            # plot_emotion_probabilities(all_class_probabilities)
            # Call the function and store the DataFrame
            df_output = plot_emotion_probabilities(all_class_probabilities, input_video)
            print("checking",df_output)
            # If you want to save the data as a list of dictionaries
            list_of_dicts = df_output.to_dict(orient="records")
            return list_of_dicts,True

        elif output_option == "video":
            # Convert list of images to video clip
            clip_with_plot = ImageSequenceClip(combined_images, fps=vid_fps)
            os.makedirs("output_videos", exist_ok=True)
            # output_video_file = os.path.splitext(input_video)[0] + "_output_video.mp4"
            output_video_file = os.path.join("output_videos", os.path.basename(input_video))
            clip_with_plot.write_videofile(output_video_file, fps=vid_fps)
            return output_video_file,True
            # clip_with_plot.ipython_display(width=900)
        else:
            print("Invalid output option. Choose either 'graph' or 'video'.")
            return "Invalid output option. Choose either 'graph' or 'video'.",False

    except Exception as e:
        return str(e),False


def plot_emotion_probabilities(all_class_probabilities, input_video):
    colors = {
        "angry": "red",
        "unconscious": "green",
        "painful": "gray",
        "happy": "yellow",
        "neutral": "purple",
        "sad": "blue",
        "surprise": "orange",
    }

    df = pd.DataFrame(all_class_probabilities)
    df = df * 100

    plt.figure(figsize=(15, 8))
    for emotion in df.columns:
        plt.plot(df[emotion], label=emotion, color=colors[emotion])

    plt.xlabel("Frame Order")
    plt.ylabel("Emotion Probability (%)")
    plt.title("Emotion Probabilities Over Time")
    plt.legend()

    # Save the plot as PNG with the same base name as the input video file
    output_file = os.path.splitext(input_video)[0] + "_output.png"
    plt.savefig(output_file)
    # plt.show()

    # Return the DataFrame for further use
    return df
def set_cache_directory():
    # Set to empty strings or skip this function if local loading works
    os.environ["XDG_CACHE_HOME"] = ""
    os.environ["HUGGINGFACE_HUB_CACHE"] = ""

if __name__ == "__main__":
    # set_cache_directory()
    path='/home/pb/hb/real-time-emotion-recognition/test_data'
    model_path = "/home/pb/hb/real-time-emotion-recognition/utils/models"

    # if os.path.exists(path):
    try:
        for i in range(len(os.listdir(path))):
            process_video(os.path.join(path,os.listdir(path)[i]), output_option='video')
    except Exception as e:
        print(str(e))
        # process_video(path, output_option='video')
    # else:
    #     print("File does not exist")
    # process_video('/home/pb/hb/real-time-emotion-recognition/test_data/arabic_emo.mp4', output_option='video')  # Change the parameters accordingly
