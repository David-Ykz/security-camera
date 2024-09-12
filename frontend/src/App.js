import logo from './logo.svg';
import './App.css';
import React, { useEffect, useRef } from 'react';



function App() {
  const videoRef = useRef();

  useEffect(() => {
    videoRef.current.src = 'http://192.168.50.179:5000/video_feed';
  }, []);
  
  function onButtonClick() {
    fetch('http://192.168.50.179:5000/test')
    .then((res) => res.text())         // Change to .json() if the response is JSON
    .then((data) => {
      console.log(data);               // Store the response in the state
    })
    .catch((err) => {
      console.error("Error fetching data: ", err);
    });
  }


  return (
    <div className="App">
      <h1>Camera Feed</h1>
      <img ref={videoRef} alt="Video feed" />
      
      <button onClick={onButtonClick}>
        test
      </button>
    </div>)
}

export default App;
