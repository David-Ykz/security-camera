import logo from './logo.svg';
import './App.css';
import React, { useState, useEffect, useRef } from 'react';



function App() {
const videoRef = useRef();
const [password, setPassword] = useState("");
const [displayVideo, setDisplayVideo] = useState(false);

useEffect(() => {
	if (videoRef.current != null) {
		videoRef.current.src = "";
	}
}, []);

function verifyToken() {
	fetch('https://knee-hoped-confusion-liberia.trycloudflare.com/authenticate', {
		method: 'POST',
		headers: {
		'Content-Type': 'application/json',
		},
		body: JSON.stringify({ token: password }),
	})
	.then(response => response.json())
	.then(data => {
		videoRef.current.src = data.videoKey;
		if (data.display === "true") {
			setDisplayVideo(data.display);			
		}
	})
}

return (
	<div className="App">
	<h1>Camera Feed</h1>
	{displayVideo ? 
		<img ref={videoRef} alt="Video feed" />
		:
		<div>
			<img ref={videoRef} alt="Video feed" style={{display: "none"}} />
		</div>
	}
	<br />
	<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Enter Password" />
	<button onClick={verifyToken}>
		Enter password
	</button>


	</div>)
}

export default App;
