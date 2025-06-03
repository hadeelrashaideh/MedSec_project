import React, { useState, useEffect } from "react";
import Layout from "../../components/Layout";
import Button from "../../components/Button";
import Spinner from "../../components/Spinner";
import { usePatients } from "../../context/PatientContext";
import { useAuth } from "../../context/AuthContext";

const SearchPatient = () => {
  const [patientId, setPatientId] = useState("");
  const [searchError, setSearchError] = useState("");
  const [patient, setPatient] = useState(null);
  const [isImageEnlarged, setIsImageEnlarged] = useState(false);
  const [imageBlob, setImageBlob] = useState(null);
  const {
    findPatientById,
    loading,
    fetchAuthenticatedImage,
    imageLoading,
    imageError,
    encryptionStatus,
  } = usePatients();
  const { tokens } = useAuth();

  // Add responsive state
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const isMobile = windowWidth < 768;
  const isTablet = windowWidth >= 768 && windowWidth < 1024;

  // Handle window resize for responsiveness
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Fetch image with authentication when patient changes and has an image_url
  useEffect(() => {
    if (patient?.image_url) {
      getPatientImage(patient.image_url);
    } else {
      setImageBlob(null);
    }
  }, [patient]);

  // Function to fetch the image with authentication and decryption if needed
  const getPatientImage = async (url) => {
    const imageUrl = await fetchAuthenticatedImage(url);
    if (imageUrl) {
      setImageBlob(imageUrl);
    }
  };

  // Cleanup object URL when component unmounts or when imageBlob changes
  useEffect(() => {
    return () => {
      if (imageBlob) {
        URL.revokeObjectURL(imageBlob);
      }
    };
  }, [imageBlob]);

  const handleSearch = async (e) => {
    e.preventDefault();

    // Clear previous results
    setPatient(null);
    setImageBlob(null);
    setSearchError("");

    if (!patientId.trim()) {
      setSearchError("Please enter a patient ID");
      return;
    }

    try {
      // Make a fresh API request every time
      const foundPatient = await findPatientById(patientId);

      if (!foundPatient) {
        setSearchError("No patient found with this ID");
        setPatient(null);
      } else {
        setPatient(foundPatient);
        setSearchError("");
      }
    } catch (err) {
      console.error("Error searching for patient:", err);
      setSearchError("An error occurred while searching");
    }
  };

  // ==== STYLES ====

  // Page container styles
  const pageContainerStyle = {
    maxWidth: isMobile ? "100%" : isTablet ? "90%" : "1100px",
    margin: "0 auto",
    padding: isMobile ? "16px" : isTablet ? "24px" : "32px",
    width: "100%",
    boxSizing: "border-box",
  };

  // Header section styles
  const headerSectionStyle = {
    marginBottom: isMobile ? "24px" : "32px",
  };

  // Header title styles
  const headerTitleStyle = {
    fontSize: isMobile ? "20px" : isTablet ? "24px" : "28px",
    fontWeight: "700",
    color: "#1E293B",
    borderLeft: "4px solid #2563EB",
    paddingLeft: "12px",
    marginBottom: "8px",
  };

  // Header description styles
  const headerDescriptionStyle = {
    fontSize: isMobile ? "14px" : "16px",
    color: "#64748B",
    marginTop: "8px",
  };

  // Search form card styles
  const searchFormCardStyle = {
    backgroundColor: "#ffffff",
    borderRadius: "12px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.05)",
    padding: isMobile ? "16px" : "24px",
    marginBottom: "24px",
    border: "1px solid #E2E8F0",
  };

  // Form styles
  const formStyle = {
    display: "flex",
    flexDirection: isMobile ? "column" : "row",
    gap: "16px",
    alignItems: isMobile ? "stretch" : "flex-end",
    width: "100%",
  };

  // Form group styles
  const formGroupStyle = {
    flexGrow: 1,
    position: "relative",
  };

  // Label styles
  const labelStyle = {
    display: "block",
    marginBottom: "8px",
    fontWeight: "500",
    fontSize: isMobile ? "14px" : "15px",
    color: "#334155",
  };

  // Input styles
  const inputStyle = {
    width: "100%",
    padding: "12px 16px",
    fontSize: isMobile ? "14px" : "15px",
    borderRadius: "8px",
    border: searchError ? "1px solid #EF4444" : "1px solid #CBD5E1",
    backgroundColor: "#FFFFFF",
    boxSizing: "border-box",
    outline: "none",
    transition: "border-color 0.2s ease, box-shadow 0.2s ease",
  };

  // Input focus/blur handlers
  const focusStyle = (e) => {
    e.target.style.borderColor = "#3B82F6";
    e.target.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.1)";
  };

  const blurStyle = (e) => {
    if (searchError) {
      e.target.style.borderColor = "#EF4444";
      e.target.style.boxShadow = "none";
    } else {
      e.target.style.borderColor = "#CBD5E1";
      e.target.style.boxShadow = "none";
    }
  };

  // Error message styles
  const errorMessageStyle = {
    color: "#EF4444",
    fontSize: "13px",
    marginTop: "4px",
    position: isMobile ? "static" : "absolute",
    paddingBottom: isMobile ? "4px" : "0",
  };

  // Button container styles
  const buttonContainerStyle = {
    display: "flex",
    alignItems: "center",
  };

  // Search button icon styles
  const searchIconStyle = {
    width: "16px",
    height: "16px",
    marginRight: "8px",
  };

  // Patient card container styles
  const patientCardStyle = {
    backgroundColor: "#FFFFFF",
    borderRadius: "12px",
    overflow: "hidden",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.08)",
    border: "1px solid #E2E8F0",
    marginBottom: "24px",
  };

  // Patient card flex container
  const patientCardFlexStyle = {
    display: "flex",
    flexDirection: isMobile ? "column" : "row",
  };

  // Patient image container styles
  const patientImageContainerStyle = {
    width: isMobile ? "100%" : "33.333%",
    backgroundColor: "#F8FAFC",
    cursor: patient?.image_url ? "pointer" : "default",
    position: "relative",
    overflow: "hidden",
  };

  // No image placeholder styles
  const noImagePlaceholderStyle = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: isMobile ? "200px" : "100%",
    backgroundColor: "#F1F5F9",
    padding: "24px",
  };

  // No image icon styles
  const noImageIconStyle = {
    width: "64px",
    height: "64px",
    marginBottom: "8px",
    color: "#94A3B8",
  };

  // Patient info container styles
  const patientInfoContainerStyle = {
    width: isMobile ? "100%" : "66.666%",
    padding: isMobile ? "20px" : "24px",
  };

  // Patient header styles
  const patientHeaderStyle = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: isMobile ? "flex-start" : "center",
    marginBottom: "24px",
    flexDirection: isMobile ? "column" : "row",
    gap: isMobile ? "12px" : "0",
  };

  // Patient name styles
  const patientNameStyle = {
    fontSize: isMobile ? "20px" : "24px",
    fontWeight: "700",
    color: "#1E293B",
  };

  // Patient ID badge styles
  const patientIdBadgeStyle = {
    backgroundColor: "#DBEAFE",
    color: "#1D4ED8",
    fontSize: "12px",
    padding: "4px 12px",
    borderRadius: "9999px",
    fontWeight: "500",
    display: "inline-block",
  };

  // Patient details grid styles
  const patientDetailsGridStyle = {
    display: "grid",
    gridTemplateColumns: isMobile ? "1fr" : "repeat(2, 1fr)",
    gap: "16px",
    marginBottom: "24px",
    borderBottom: "1px solid #E2E8F0",
    paddingBottom: "16px",
  };

  // Patient detail label styles
  const patientDetailLabelStyle = {
    fontSize: "13px",
    fontWeight: "500",
    color: "#64748B",
    marginBottom: "4px",
  };

  // Patient detail value styles
  const patientDetailValueStyle = {
    fontSize: isMobile ? "16px" : "18px",
    color: "#334155",
    fontWeight: "500",
  };

  // Patient results section styles
  const patientNotesSectionStyle = {
    paddingTop: "16px",
    width: isMobile ? "90%" : "100%",
  };

  // Patient results title styles
  const patientNotesTitleStyle = {
    fontSize: isMobile ? "16px" : "18px",
    fontWeight: "600",
    color: "#1E293B",
    marginBottom: "12px",
  };

  // Test results content styles
  const notesContentStyle = {
    backgroundColor: "#EFF6FF",
    color: "#1E40AF",
    padding: "16px",
    borderRadius: "8px",
    fontSize: isMobile ? "14px" : "15px",
    lineHeight: "1.5",
    marginBottom: "20px",
    minHeight: "40px",
    display: "flex",
    alignItems: "center",
  };

  // No results message styles
  const noNotesMessageStyle = {
    color: "#64748B",
    fontStyle: "italic",
  };

  // Actions container styles
  const actionsContainerStyle = {
    marginTop: "12px",
    display: "flex",
    justifyContent: isMobile ? "center" : "flex-end",
    gap: "12px",
    flexDirection: isMobile ? "column" : "row",
    width: isMobile ? "90%" : "100%",
  };

  // No patient found card styles
  const noPatientCardStyle = {
    backgroundColor: "#FFFFFF",
    padding: "24px",
    borderRadius: "12px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.05)",
    textAlign: "center",
    border: "1px solid #E2E8F0",
    marginBottom: "24px",
  };

  // No patient icon styles
  const noPatientIconStyle = {
    width: "48px",
    height: "48px",
    margin: "0 auto 16px auto",
    color: "#EF4444",
  };

  // No patient title styles
  const noPatientTitleStyle = {
    fontSize: isMobile ? "16px" : "18px",
    fontWeight: "500",
    color: "#111827",
    marginBottom: "8px",
  };

  // No patient message styles
  const noPatientMessageStyle = {
    color: "#64748B",
    marginBottom: "16px",
  };

  // Patient ID highlight styles
  const patientIdHighlightStyle = {
    fontWeight: "600",
  };

  // Patient image zoom overlay
  const imageOverlayStyle = {
    position: "fixed",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    backgroundColor: "rgba(0, 0, 0, 0.85)",
    display: isImageEnlarged ? "flex" : "none",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 1000,
    backdropFilter: "blur(4px)",
  };

  // Enlarged image container
  const enlargedImageContainerStyle = {
    position: "relative",
    maxWidth: "90%",
    maxHeight: "90%",
    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.3)",
    borderRadius: "8px",
    overflow: "hidden",
  };

  // Enlarged image style
  const enlargedImageStyle = {
    maxWidth: "100%",
    maxHeight: "90vh",
    objectFit: "contain",
  };

  // Close button for enlarged image
  const closeEnlargedImageButtonStyle = {
    position: "absolute",
    top: "12px",
    right: "12px",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    color: "white",
    borderRadius: "50%",
    width: "36px",
    height: "36px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    border: "none",
    outline: "none",
  };

  // Image zoom icon
  const zoomIconStyle = {
    position: "absolute",
    top: "12px",
    right: "12px",
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    color: "white",
    borderRadius: "50%",
    width: "32px",
    height: "32px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    zIndex: 10,
  };

  // Patient image responsive style
  const patientImageStyle = {
    width: "100%",
    height: isMobile ? "250px" : "100%",
    objectFit: "cover",
    transition: "transform 0.3s ease",
  };

  // Hover text for image zoom
  const zoomTextStyle = {
    position: "absolute",
    bottom: "12px",
    left: "0",
    right: "0",
    textAlign: "center",
    backgroundColor: "rgba(0, 0, 0, 0.6)",
    color: "white",
    padding: "8px",
    fontSize: "12px",
    opacity: 0,
    transition: "opacity 0.2s ease",
  };

  // Function to format image URL
  const formatImageUrl = (imagePath) => {
    if (!imagePath) return null;
    // If URL is already absolute with http/https, use it as is
    if (imagePath.startsWith("http")) {
      return imagePath;
    }
    // Otherwise, prefix with the API base URL
    return `http://localhost:8000${imagePath}`;
  };

  // Encryption indicator styles
  const encryptionIndicatorStyle = {
    display: "flex",
    alignItems: "center",
    fontSize: "12px",
    fontWeight: "500",
    color: encryptionStatus ? "#047857" : "#6B7280",
    backgroundColor: encryptionStatus ? "#D1FAE5" : "#F3F4F6",
    padding: "4px 8px",
    borderRadius: "4px",
    marginLeft: isMobile ? "0" : "12px",
    marginTop: isMobile ? "8px" : "0",
  };

  const encryptionIconStyle = {
    width: "14px",
    height: "14px",
    marginRight: "6px",
  };

  return (
    <Layout>
      <div style={pageContainerStyle}>
        <div style={headerSectionStyle}>
          <h1 style={headerTitleStyle}>Patient Lookup</h1>
          <div
            style={{
              display: "flex",
              alignItems: isMobile ? "flex-start" : "center",
              flexDirection: isMobile ? "column" : "row",
            }}
          >
            <p style={headerDescriptionStyle}>
              Enter the patient ID to view their details and test results
            </p>
            <div style={encryptionIndicatorStyle}>
              <svg
                style={encryptionIconStyle}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d={
                    encryptionStatus
                      ? "M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                      : "M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                  }
                />
              </svg>
              {encryptionStatus
                ? "End-to-End Encryption"
                : "Standard Encryption"}
            </div>
          </div>
        </div>

        <div style={searchFormCardStyle}>
          <form onSubmit={handleSearch} style={formStyle}>
            <div style={formGroupStyle}>
              <label style={labelStyle} htmlFor="patientId">
                Patient ID
              </label>
              <input
                id="patientId"
                type="text"
                style={inputStyle}
                value={patientId}
                onChange={(e) => {
                  setPatientId(e.target.value);
                  setSearchError("");
                }}
                placeholder="Enter patient ID (e.g., P001)"
                onFocus={focusStyle}
                onBlur={blurStyle}
              />
              {searchError && (
                <div style={errorMessageStyle}>{searchError}</div>
              )}
            </div>
            <div style={buttonContainerStyle}>
              <Button
                type="submit"
                variant="primary"
                disabled={loading}
                style={{
                  width: isMobile ? "100%" : "auto",
                  padding: "12px 20px",
                  fontSize: "15px",
                  borderRadius: "8px",
                  marginTop: isMobile && searchError ? "24px" : "0",
                  backgroundColor: "#0EA5E9",
                  borderColor: "#0EA5E9",
                }}
              >
                {loading ? (
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <Spinner size="small" color="white" />
                    <span style={{ marginLeft: "8px" }}>Searching...</span>
                  </div>
                ) : (
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: isMobile ? "center" : "flex-start",
                    }}
                  >
                    <svg
                      style={searchIconStyle}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                      />
                    </svg>
                    Search
                  </div>
                )}
              </Button>
            </div>
          </form>
        </div>

        {/* Patient Image Zoom Overlay */}
        <div
          style={imageOverlayStyle}
          onClick={() => setIsImageEnlarged(false)}
        >
          <div
            style={enlargedImageContainerStyle}
            onClick={(e) => e.stopPropagation()}
          >
            {imageBlob && (
              <img
                src={imageBlob}
                alt={patient?.name}
                style={enlargedImageStyle}
              />
            )}
            <button
              style={closeEnlargedImageButtonStyle}
              onClick={() => setIsImageEnlarged(false)}
              aria-label="Close image preview"
            >
              <svg
                width="20"
                height="20"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>

        {patient && (
          <div style={patientCardStyle}>
            <div style={patientCardFlexStyle}>
              <div
                style={patientImageContainerStyle}
                onMouseEnter={(e) => {
                  if (imageBlob) {
                    const zoomText =
                      e.currentTarget.querySelector("[data-zoom-text]");
                    if (zoomText) zoomText.style.opacity = "1";
                  }
                }}
                onMouseLeave={(e) => {
                  if (imageBlob) {
                    const zoomText =
                      e.currentTarget.querySelector("[data-zoom-text]");
                    if (zoomText) zoomText.style.opacity = "0";
                  }
                }}
              >
                {imageLoading ? (
                  <div style={noImagePlaceholderStyle}>
                    <Spinner size="medium" color="#64748B" />
                    <p style={{ marginTop: "16px", color: "#64748B" }}>
                      Loading image...
                    </p>
                  </div>
                ) : imageBlob ? (
                  <div style={{ height: "100%" }}>
                    <img
                      src={imageBlob}
                      alt={patient.name}
                      style={patientImageStyle}
                      onClick={() => setIsImageEnlarged(true)}
                    />
                    <div
                      style={zoomIconStyle}
                      onClick={() => setIsImageEnlarged(true)}
                    >
                      <svg
                        width="18"
                        height="18"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"
                        />
                      </svg>
                    </div>
                    <div style={zoomTextStyle} data-zoom-text="true">
                      Click to enlarge image
                    </div>
                    {encryptionStatus && (
                      <div
                        style={{
                          position: "absolute",
                          bottom: "12px",
                          right: "12px",
                          backgroundColor: "rgba(0, 0, 0, 0.6)",
                          color: "white",
                          padding: "4px 8px",
                          borderRadius: "4px",
                          fontSize: "12px",
                          display: "flex",
                          alignItems: "center",
                        }}
                      >
                        <svg
                          style={{
                            width: "12px",
                            height: "12px",
                            marginRight: "4px",
                          }}
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                          />
                        </svg>
                        Encrypted
                      </div>
                    )}
                  </div>
                ) : imageError ? (
                  <div style={noImagePlaceholderStyle}>
                    <svg
                      style={noImageIconStyle}
                      fill="none"
                      stroke="#EF4444"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <p style={{ color: "#EF4444" }}>Error loading image</p>
                  </div>
                ) : (
                  <div style={noImagePlaceholderStyle}>
                    <svg
                      style={noImageIconStyle}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                      />
                    </svg>
                    <p>No image available</p>
                  </div>
                )}
              </div>

              <div style={patientInfoContainerStyle}>
                <div style={patientHeaderStyle}>
                  <h2 style={patientNameStyle}>{patient.name}</h2>
                  <div style={patientIdBadgeStyle}>
                    Patient ID: {patient.id}
                  </div>
                </div>

                <div style={patientDetailsGridStyle}>
                  <div>
                    <h3 style={patientDetailLabelStyle}>Age</h3>
                    <p style={patientDetailValueStyle}>{patient.age} years</p>
                  </div>
                  {patient.gender && (
                    <div>
                      <h3 style={patientDetailLabelStyle}>Gender</h3>
                      <p style={patientDetailValueStyle}>{patient.gender}</p>
                    </div>
                  )}
                  {patient.contactNumber && (
                    <div>
                      <h3 style={patientDetailLabelStyle}>Contact Number</h3>
                      <p style={patientDetailValueStyle}>
                        {patient.contactNumber}
                      </p>
                    </div>
                  )}
                  {patient.email && (
                    <div>
                      <h3 style={patientDetailLabelStyle}>Email</h3>
                      <p style={patientDetailValueStyle}>{patient.email}</p>
                    </div>
                  )}
                </div>

                <div style={patientNotesSectionStyle}>
                  <h3 style={patientNotesTitleStyle}>Notes</h3>
                  {patient.note ? (
                    <div style={notesContentStyle}>{patient.note}</div>
                  ) : (
                    <p style={noNotesMessageStyle}>No notes available</p>
                  )}
                </div>

                <div style={actionsContainerStyle}>
                  <Button
                    variant="secondary"
                    size="small"
                    onClick={() => {
                      setPatient(null);
                      setPatientId("");
                      setImageBlob(null);
                    }}
                    style={{
                      padding: "10px 16px",
                      fontSize: "14px",
                      borderRadius: "8px",
                      width: isMobile ? "100%" : "auto",
                      backgroundColor: "#F3E8FF",
                      color: "#9333EA",
                      border: "1px solid #E9D5FF",
                    }}
                  >
                    Clear Results
                  </Button>
                  {imageBlob && (
                    <Button
                      variant="primary"
                      size="small"
                      onClick={() => setIsImageEnlarged(true)}
                      style={{
                        padding: "10px 16px",
                        fontSize: "14px",
                        borderRadius: "8px",
                        width: isMobile ? "100%" : "auto",
                        backgroundColor: "#0EA5E9",
                        borderColor: "#0EA5E9",
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          justifyContent: isMobile ? "center" : "flex-start",
                        }}
                      >
                        <svg
                          style={{
                            width: "16px",
                            height: "16px",
                            marginRight: "6px",
                          }}
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                          />
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                          />
                        </svg>
                        View Image
                      </div>
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {!patient && !loading && patientId && searchError && (
          <div style={noPatientCardStyle}>
            <div style={{ color: "#EF4444" }}>
              <svg
                style={noPatientIconStyle}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <h3 style={noPatientTitleStyle}>No Results Found</h3>
            <p style={noPatientMessageStyle}>
              We couldn't find a patient with the ID:{" "}
              <span style={patientIdHighlightStyle}>{patientId}</span>
            </p>
            <Button
              variant="secondary"
              size="small"
              onClick={() => {
                setPatientId("");
                setSearchError("");
              }}
              style={{
                padding: "10px 16px",
                fontSize: "14px",
                borderRadius: "8px",
                width: isMobile ? "100%" : "auto",
                maxWidth: "200px",
                margin: isMobile ? "0 auto" : "0",
                backgroundColor: "#F3E8FF",
                color: "#9333EA",
                border: "1px solid #E9D5FF",
              }}
            >
              Try Again
            </Button>
          </div>
        )}
      </div>
    </Layout>
  );
};

export default SearchPatient;
