import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <h1>Camera Feed</h1>
      <video
        src="http://192.168.50.179:5000/video_feed"
        width="600"
        height="400"
        autoPlay
      />
    </div>)
}

export default App;
