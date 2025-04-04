
from prometheus_client import Counter, Gauge, Histogram

# Classification request metrics
CLASSIFICATION_REQUESTS = Counter(
    'classification_requests_total',
    'Total number of classification requests',
    ['status']  # success, error
)

# Classification latency metrics
CLASSIFICATION_LATENCY = Histogram(
    'classification_latency_seconds',
    'Time taken to classify an image',
    buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0]
)

# Predicted class distribution metrics
CLASSIFICATION_DISTRIBUTION = Counter(
    'predicted_class_distribution_total',
    'Distribution of predicted classes',
    ['predicted_class']  # asphalt, chip-sealed, gravel, uncertain
)

# Queue metrics
QUEUE_MESSAGES_PUBLISHED = Counter(
    'queue_messages_published_total',
    'Number of messages published to RabbitMQ'
)

# System metrics
MODEL_READY = Gauge(
    'model_ready',
    'Indicates if the model is ready for inference'
)

# Process-specific metrics
PROCESS_MODEL_STATUS = Gauge(
    'process_model_status',
    'Status of model in each process',
    ['process_id']  # Will be set to the process PID
)

ACTIVE_CLASSIFICATIONS = Gauge(
    'active_classifications',
    'Number of currently running classifications',
    ['process_id']
)

PROCESS_LAST_HEARTBEAT = Gauge(
    'process_last_heartbeat_timestamp',
    'Last heartbeat timestamp from each process',
    ['process_id']
)

# Total active processes
ACTIVE_PROCESSES = Gauge(
    'active_processes',
    'Number of active classification processes'
)

def record_classification_result(predicted_class: str):
    """Record the predicted class distribution"""
    CLASSIFICATION_DISTRIBUTION.labels(
        predicted_class=predicted_class.lower()
    ).inc()

def clear_process_metrics(process_id: str):
    """Clear all metrics for a specific process"""
    PROCESS_MODEL_STATUS.labels(process_id=process_id).set(-1)
    ACTIVE_CLASSIFICATIONS.labels(process_id=process_id).set(0)
    PROCESS_LAST_HEARTBEAT.labels(process_id=process_id).set(0)

