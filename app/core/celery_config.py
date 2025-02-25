from celery import Celery
from celery.schedules import crontab
import os

celery = Celery(
    __name__,
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    include=["app.batch"]
)

celery.conf.update(
    result_expires=3600,
    task_track_started=True
)

celery.conf.beat_schedule = {
    "run-every-1-minute": {
        "task": "app.batch.process_images_from_dir",
        "schedule": crontab(minute="*/1"),
        "args": ("/Users/ayodele/Documents/pictures",)
    }
}

if __name__ == "__main__":
    celery.start()