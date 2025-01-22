
import uvicorn
from dotenv import load_dotenv
from src.core.settings import settings

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "src.service.service:app",  # Change this line
        host=settings.HOST,
        port=settings.PORT,
        log_level="debug",
        reload=True
    )