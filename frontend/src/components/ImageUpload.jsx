import "../styles/ImageUpload.css";
function ImageUpload() {
  return (
    <div className="upload-box">

      <input type="file" />

      <br /><br />

      <button>Detect Image</button>

    </div>
  );
}

export default ImageUpload;