from fastapi import FastAPI,File, UploadFile, Request,HTTPException
from fastapi.responses import HTMLResponse, JSONResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import shutil
from utils.emotions import process_video
from fastapi.middleware.cors import CORSMiddleware
import asyncio
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
CHUNK_DIR = os.path.join("static", "chunks")
chunk_count = 0
m3u8_content = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:5\n#EXT-X-MEDIA-SEQUENCE:0\n"

@app.get("/")
async def handshake():
    return "Connection established"


@app.post("/video_upload/")
async def video_upload(file: UploadFile = File(...),output_option:str = "graph"):
    try:
        file_path = os.path.join("static", file.filename)
        if not file.filename.endswith(".mp4"):
            raise HTTPException(status_code=400, detail="Only mp4 files are allowed")
        # Save the uploaded file to the 'static' folder
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        output=None
        response=None
        if output_option in ["graph","video"]:
            output,response = process_video(file_path,output_option)
        if response:
            if output_option == "graph":
                return JSONResponse(status_code=200,content=output)
            else:
                return JSONResponse(status_code=200,content={"file_path":output})
        else:
            raise HTTPException(status_code=400, detail="Error processing video")
        
        return {"filename": file.filename, "file_path": file_path}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error", "Error": str(e)})

@app.get("/livestream/")
async def livestream(request: Request):
    return templates.TemplateResponse("live.html", {"request": request})

@app.post("/upload-chunk")
async def upload_chunk(video: UploadFile = File(...)):
    global chunk_count,m3u8_content
    if not os.path.exists(CHUNK_DIR):
        os.makedirs(CHUNK_DIR)
    
    
    
    chunk_filename = f"chunk_{chunk_count}.mp4"
    chunk_path = os.path.join(CHUNK_DIR, chunk_filename)
    
    # Save the chunk
    with open(chunk_path, "wb") as buffer:
        content = await video.read()
        buffer.write(content)
    response=None
    output,response = process_video(chunk_path,output_option="video")
    # # Update m3u8 content
    # m3u8_content += f"#EXTINF:5.0,\n{chunk_filename}\n"
    chunk_count += 1
    
    # # Update m3u8 file
    # with open(os.path.join(CHUNK_DIR, "playlist.m3u8"), "w") as m3u8_file:
    #     m3u8_file.write(m3u8_content)
    
    return {"message": "Chunk uploaded successfully"}

@app.get("/playlist.m3u8")
async def get_playlist():
    return FileResponse(os.path.join(CHUNK_DIR, "playlist.m3u8"), media_type="application/vnd.apple.mpegurl")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)