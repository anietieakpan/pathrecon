// Global variables
const videoFeed = document.getElementById('video-feed');
const platesUl = document.getElementById('plates-ul');
let currentVideoPath = '';

// Database Info
function fetchDBInfo() {
    fetch('/db_info')
        .then(response => response.json())
        .then(data => {
            document.getElementById('db-info').innerHTML = `
                <h3>Database Info</h3>
                <p>URL: ${data.url}</p>
                <p>Organization: ${data.org}</p>
                <p>Bucket: ${data.bucket}</p>
            `;
        })
        .catch(error => console.error('Error:', error));
}

// Video handling functions
function uploadVideo() {
    const input = document.getElementById('video-input');
    const file = input.files[0];
    if (!file) {
        alert('Please select a video file');
        return;
    }

    const maxSize = 300 * 1024 * 1024; // 100 MB limit
    if (file.size > maxSize) {
        alert('File is too large. Please select a file smaller than 100 MB.');
        return;
    }

    const formData = new FormData();
    formData.append('video', file);

    fetch('/upload_video', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            currentVideoPath = data.filepath;
            alert('Video uploaded successfully. You can now start the video.');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while uploading the video. Please try again.');
        });
}

function startVideo() {
    if (!currentVideoPath) {
        alert('Please upload a video first');
        return;
    }

    fetch('/start_video', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ videoPath: currentVideoPath })
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            videoFeed.src = '/video_feed';
        })
        .catch(error => console.error('Error:', error));
}

function stopVideo() {
    fetch('/stop_video')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            videoFeed.src = '';
        })
        .catch(error => console.error('Error:', error));
}

// Detection functions
function refreshPlates() {
    fetch('/detected_plates')
        .then(response => response.json())
        .then(plates => {
            platesUl.innerHTML = '';
            plates.forEach(plate => {
                const li = createDetectionListItem(plate);
                platesUl.appendChild(li);
            });
        })
        .catch(error => console.error('Error:', error));
}

function createDetectionListItem(detection) {
    const li = document.createElement('li');
    let content = `Plate: ${detection.text} (Confidence: ${detection.confidence.toFixed(2)})`;

    // Add vehicle details if available
    if (detection.vehicle_details) {
        const details = detection.vehicle_details;
        content += `
            <div class="vehicle-details">
                <span>Make: ${details.make || 'Unknown'}</span>
                <span>Model: ${details.model || 'Unknown'}</span>
                <span>Color: ${details.color || 'Unknown'}</span>
                <span>Type: ${details.type || 'Unknown'}</span>
            </div>
        `;
    }

    li.innerHTML = content;
    return li;
}

// Image processing functions
function processImage() {
    const input = document.getElementById('image-input');
    const file = input.files[0];
    if (!file) {
        alert('Please select an image file');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    fetch('/process_image', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            const img = document.getElementById('processed-image');
            img.src = 'data:image/jpeg;base64,' + data.image;

            displayDetections(data.detections);
        })
        .catch(error => console.error('Error:', error));
}

function displayDetections(detections) {
    platesUl.innerHTML = '';
    detections.forEach(detection => {
        const li = createDetectionListItem(detection);
        platesUl.appendChild(li);
    });
}

// Camera functions
function startCamera() {
    fetch('/start_camera')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            videoFeed.src = '/camera_feed';
        })
        .catch(error => console.error('Error:', error));
}

function stopCamera() {
    fetch('/stop_camera')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            videoFeed.src = '';
        })
        .catch(error => console.error('Error:', error));
}

// Configuration functions
function updateConfig() {
    const config = {
        FRAME_SKIP: parseInt(document.getElementById('frame-skip').value),
        RESIZE_WIDTH: parseInt(document.getElementById('resize-width').value),
        RESIZE_HEIGHT: parseInt(document.getElementById('resize-height').value),
        CONFIDENCE_THRESHOLD: parseFloat(document.getElementById('confidence-threshold').value),
        MAX_DETECTIONS_PER_FRAME: parseInt(document.getElementById('max-detections').value),
        PROCESS_EVERY_N_SECONDS: parseFloat(document.getElementById('process-interval').value)
    };

    fetch('/update_config', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(config)
    })
        .then(response => response.json())
        .then(data => alert(data.message))
        .catch(error => console.error('Error:', error));
}

function loadConfig() {
    fetch('/get_config')
        .then(response => response.json())
        .then(config => {
            document.getElementById('frame-skip').value = config.FRAME_SKIP;
            document.getElementById('resize-width').value = config.RESIZE_WIDTH;
            document.getElementById('resize-height').value = config.RESIZE_HEIGHT;
            document.getElementById('confidence-threshold').value = config.CONFIDENCE_THRESHOLD;
            document.getElementById('max-detections').value = config.MAX_DETECTIONS_PER_FRAME;
            document.getElementById('process-interval').value = config.PROCESS_EVERY_N_SECONDS;
        })
        .catch(error => console.error('Error:', error));
}

// Vehicle-specific functions
function loadVehicleData() {
    // Load makes/models
    fetch('/vehicle/makes')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                populateVehicleSelects(data.makes);
            }
        })
        .catch(error => console.error('Error:', error));
}

function populateVehicleSelects(makes) {
    const makeSelect = document.getElementById('make-select');
    makes.forEach(make => {
        const option = document.createElement('option');
        option.value = make.vehicle_make;
        option.textContent = make.vehicle_make;
        makeSelect.appendChild(option);
    });
}

// Initialize
window.onload = function () {
    loadConfig();
    fetchDBInfo();
    loadVehicleData();
    // Start auto-refresh
    setInterval(refreshPlates, 5000);
};