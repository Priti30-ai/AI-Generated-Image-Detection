import { NavLink, Link } from "react-router-dom";
import "../styles/Navbar.css";

function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">
        <span className="navbar-logo">🔍</span>
        <h2 className="navbar-title">AI Image Detector</h2>
      </Link>
      <ul className="navbar-nav">
        <li>
          <NavLink
            to="/"
            end
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            Detector
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/about"
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            About & Architecture
          </NavLink>
        </li>
      </ul>
    </nav>
  );
}

export default Navbar;