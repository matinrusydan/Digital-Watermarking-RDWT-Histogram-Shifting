import os

class Config:
    SECRET_KEY = 'your-secret-key-here'  # Change this in production
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    RESULTS_FOLDER = 'results'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

    @classmethod
    def init_app(cls, app):
        for folder in [cls.UPLOAD_FOLDER, cls.RESULTS_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
