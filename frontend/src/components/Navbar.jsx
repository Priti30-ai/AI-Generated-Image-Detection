import { Link } from "react-router-dom";
import "../styles/Navbar.css";

function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">
        <span className="navbar-logo">🔍</span>
        <h2 className="navbar-title">AI Image Detector</h2>
      </Link>
    </nav>
  );
}

export default Navbar;