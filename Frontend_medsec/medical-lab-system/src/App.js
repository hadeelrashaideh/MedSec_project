import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";

// Context Providers
import { AuthProvider } from "./context/AuthContext";
import { PatientProvider } from "./context/PatientContext";

// Components
import ProtectedRoute from "./components/ProtectedRoute";

// Pages
import Home from "./pages/Home";
import Login from "./pages/Login";
import AddPatient from "./pages/lab/AddPatient";
import SearchPatient from "./pages/doctor/SearchPatient";
import NotFound from "./pages/NotFound";

// CSS
import "./App.css";

function App() {
  return (
    <AuthProvider>
      <PatientProvider>
        <Router>
          <Routes>
            {/* Redirect root to login */}
            <Route path="/" element={<Navigate to="/login" replace />} />

            {/* Public Routes */}
            <Route path="/home" element={<Home />} />
            <Route path="/login" element={<Login />} />

            {/* Lab Routes */}
            <Route element={<ProtectedRoute requiredRole="lab" />}>
              <Route path="/lab/add-patient" element={<AddPatient />} />
            </Route>

            {/* Doctor Routes */}
            <Route element={<ProtectedRoute requiredRole="doctor" />}>
              <Route path="/doctor/search" element={<SearchPatient />} />
            </Route>

            {/* 404 Route */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Router>
      </PatientProvider>
    </AuthProvider>
  );
}

export default App;
