import React from "react";
import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const ProtectedRoute = ({ requiredRole }) => {
  const { user, isAuthenticated } = useAuth();
  const location = useLocation();

  console.log("ProtectedRoute check:", {
    isAuthenticated,
    user,
    requiredRole,
    currentPath: location.pathname,
  });

  // If not authenticated, redirect to login
  if (!isAuthenticated) {
    console.log("Not authenticated, redirecting to login");
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // If role is specified and user doesn't have the required role, redirect to home
  if (requiredRole && user.role !== requiredRole) {
    console.log(
      `Role mismatch: required=${requiredRole}, user=${user.role}, redirecting to home`
    );
    return <Navigate to="/home" replace />;
  }

  console.log("Access granted to protected route");
  // Render the outlet (child routes)
  return <Outlet />;
};

export default ProtectedRoute;
