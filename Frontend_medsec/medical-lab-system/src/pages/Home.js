import React from "react";
import { Link } from "react-router-dom";
import Layout from "../components/Layout";
import { useAuth } from "../context/AuthContext";

const Home = () => {
  const { isAuthenticated, isLab, isDoctor } = useAuth();

  return (
    <Layout>
      <div className="max-w-4xl mx-auto text-center py-10">
        <h1 className="text-4xl font-bold text-blue-600 mb-6">
          Welcome to MedSec System
        </h1>

        <p className="text-xl text-gray-600 mb-8">
          A comprehensive solution for medical laboratories and doctors to
          manage patient results
        </p>

        <div className="bg-white p-8 rounded-lg shadow-md">
          {isAuthenticated ? (
            <div className="space-y-6">
              <div className="bg-blue-50 p-4 rounded-md">
                <h2 className="text-2xl font-bold text-gray-800 mb-2">
                  {isLab ? "Laboratory Dashboard" : "Doctor Dashboard"}
                </h2>
                <p className="text-gray-600 mb-4">
                  {isLab
                    ? "You can add new patients and their test results"
                    : "You can search for patient records and view their test results"}
                </p>

                {isLab ? (
                  <Link
                    to="/lab/add-patient"
                    className="inline-block px-6 py-3 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 transition-colors"
                  >
                    Add New Patient
                  </Link>
                ) : (
                  <Link
                    to="/doctor/search"
                    className="inline-block px-6 py-3 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 transition-colors"
                  >
                    Search Patients
                  </Link>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-50 p-6 rounded-md">
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">
                    Quick Guide
                  </h3>
                  <ul className="text-left text-gray-600 space-y-2">
                    {isLab ? (
                      <>
                        <li>• Add new patient information</li>
                        <li>• Upload patient photos</li>
                        <li>• Record test results</li>
                      </>
                    ) : (
                      <>
                        <li>• Search patients by ID</li>
                        <li>• View patient details</li>
                        <li>• Check test results</li>
                      </>
                    )}
                  </ul>
                </div>

                <div className="bg-gray-50 p-6 rounded-md">
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">
                    System Information
                  </h3>
                  <ul className="text-left text-gray-600 space-y-2">
                    <li>• Version: 1.0.0</li>
                    <li>• Role: {isLab ? "Laboratory" : "Doctor"}</li>
                    <li>• Status: Connected</li>
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">
                Please Log In
              </h2>
              <p className="text-gray-600 mb-6">
                Log in to access the MedSec system and manage patient records
              </p>

              <div className="flex justify-center space-x-4">
                <Link
                  to="/login"
                  className="px-6 py-3 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 transition-colors"
                >
                  Sign In Now
                </Link>
              </div>

              <div className="mt-8 border-t border-gray-200 pt-6">
                <h3 className="text-xl font-semibold text-gray-800 mb-4">
                  System Features
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
                  <div>
                    <h4 className="font-semibold text-blue-600 mb-2">
                      For Laboratories
                    </h4>
                    <ul className="text-gray-600 space-y-2">
                      <li>• Patient registration</li>
                      <li>• Test result upload</li>
                      <li>• Secure data management</li>
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-semibold text-blue-600 mb-2">
                      For Doctors
                    </h4>
                    <ul className="text-gray-600 space-y-2">
                      <li>• Quick patient lookup</li>
                      <li>• Comprehensive test results</li>
                      <li>• Patient history access</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
};

export default Home;
