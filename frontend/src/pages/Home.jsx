import Navbar from "../components/Navbar";
import ImageUpload from "../components/ImageUpload";
import Footer from "../components/Footer";
import "../styles/Home.css";

function Home() {
  return (
    <>
      <Navbar />

      <div className="home">

        <h1>AI Generated Image Detection</h1>

        <p>
          Upload an image to check whether it is Real or AI Generated.
        </p>

        <ImageUpload />

      </div>

      <Footer />
    </>
  );
}

export default Home;