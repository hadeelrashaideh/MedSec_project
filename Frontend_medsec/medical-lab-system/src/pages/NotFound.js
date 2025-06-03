import React from "react";
import { Link } from "react-router-dom";
import Layout from "../components/Layout";

const NotFound = () => {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto text-center py-16">
        <div className="mb-6">
          <span className="text-6xl font-bold text-blue-600">404</span>
        </div>

        <h1 className="text-3xl font-semibold text-gray-800 mb-4">
          Page Not Found
        </h1>

        <p className="text-xl text-gray-600 mb-8">
          The page you are looking for doesn't exist or has been moved.
        </p>

        <Link
          to="/"
          className="inline-block px-6 py-3 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 transition-colors"
        >
          Return to Home
        </Link>
      </div>
    </Layout>
  );
};

export default NotFound;
