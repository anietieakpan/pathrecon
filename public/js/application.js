/* public/css/styles.css */

/* Base styles */
body {
    font - family: Arial, sans - serif;
    text - align: center;
    max - width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

h1, h2, h3 {
    color: #333;
}

/* Container and layout */
.container {
    display: grid;
    grid - template - columns: 2fr 1fr;
    gap: 20px;
}

.main - content {
    grid - column: 1;
}

.sidebar {
    grid - column: 2;
}

/* Video and image sections */
#video - container,
    #image - container {
    max - width: 100 %;
    margin: 20px auto;
    border: 2px solid #333;
    border - radius: 4px;
    overflow: hidden;
}

#video - feed,
    #processed - image {
    width: 100 %;
    max - width: 100 %;
    display: block;
}

/* Control sections */
.controls {
    margin: 20px 0;
    padding: 10px;
    background - color: #f5f5f5;
    border - radius: 4px;
}

button {
    padding: 10px 20px;
    margin: 0 10px;
    font - size: 16px;
    background - color: #4CAF50;
    color: white;
    border: none;
    border - radius: 4px;
    cursor: pointer;
    transition: background - color 0.3s;
}

button:hover {
    background - color: #45a049;
}

/* Upload sections */
#image - upload,
    #video - upload {
    margin: 20px 0;
    padding: 15px;
    border: 1px solid #ddd;
    border - radius: 4px;
    background - color: #f9f9f9;
}

input[type = "file"] {
    margin: 10px 0;
}

/* Detection list */
#plates - list {
    margin - top: 20px;
    text - align: left;
    max - height: 300px;
    overflow - y: auto;
    border: 1px solid #ddd;
    padding: 10px;
    border - radius: 4px;
}

.detection - card {
    border: 1px solid #ddd;
    margin: 10px 0;
    padding: 10px;
    border - radius: 4px;
    background - color: #fff;
}

.vehicle - details {
    margin: 10px 0 5px 15px;
    padding - left: 10px;
    border - left: 3px solid #4CAF50;
    font - size: 0.9em;
}

/* Configuration form */
#config - form {
    margin: 20px 0;
    padding: 15px;
    border: 1px solid #ddd;
    border - radius: 4px;
    background - color: #f9f9f9;
    text - align: left;
}

#config - form label {
    display: inline - block;
    width: 200px;
    margin: 5px 0;
}

#config - form input {
    margin: 5px;
    padding: 5px;
    border: 1px solid #ddd;
    border - radius: 3px;
    width: 100px;
}

/* Database info section */
#db - info {
    margin: 20px 0;
    padding: 15px;
    border: 1px solid #ddd;
    border - radius: 4px;
    background - color: #f9f9f9;
}

/* Confidence indicators */
.confidence - high {
    color: #4CAF50;
}

.confidence - medium {
    color: #FFC107;
}

.confidence - low {
    color: #f44336;
}

/* Vehicle search section */
.vehicle - search {
    margin: 20px 0;
    padding: 15px;
    border: 1px solid #ddd;
    border - radius: 4px;
}

.vehicle - search select {
    margin: 5px;
    padding: 5px;
    border: 1px solid #ddd;
    border - radius: 3px;
    width: 150px;
}

/* Responsive design */
@media(max - width: 1000px) {
    .container {
        grid - template - columns: 1fr;
    }
    
    .main - content, .sidebar {
        grid - column: 1;
    }
}