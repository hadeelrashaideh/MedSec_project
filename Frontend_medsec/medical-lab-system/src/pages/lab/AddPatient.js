import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Layout from "../../components/Layout";
import Button from "../../components/Button";
import Spinner from "../../components/Spinner";
import { usePatients } from "../../context/PatientContext";

const AddPatient = () => {
  // Form state
  const [formData, setFormData] = useState({
    name: "",
    age: "",
    id: "",
    notes: "",
  });
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [formErrors, setFormErrors] = useState({});
  const [success, setSuccess] = useState(false);
  const [addedPatientId, setAddedPatientId] = useState("");

  // Responsive state
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const isMobile = windowWidth < 768;
  const isTablet = windowWidth >= 768 && windowWidth < 1024;

  const { addPatient, loading } = usePatients();
  const navigate = useNavigate();

  // Handle window resize for responsiveness
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Input change handler
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));

    // Clear error when field is edited
    if (formErrors[name]) {
      setFormErrors((prevErrors) => ({
        ...prevErrors,
        [name]: null,
      }));
    }
  };

  // Image change handler
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Check file type and size
    if (!file.type.match("image.*")) {
      setFormErrors({
        ...formErrors,
        image: "Please select an image file",
      });
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      // 5MB limit
      setFormErrors({
        ...formErrors,
        image: "Image must be less than 5MB",
      });
      return;
    }

    setImage(file);
    setFormErrors({
      ...formErrors,
      image: null,
    });

    // Create preview URL
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);
  };

  // Form validation
  const validateForm = () => {
    const errors = {};

    if (!formData.name.trim()) {
      errors.name = "Name is required";
    }

    if (!formData.age) {
      errors.age = "Age is required";
    } else if (isNaN(formData.age) || formData.age <= 0) {
      errors.age = "Age must be a positive number";
    }

    if (!formData.id.trim()) {
      errors.id = "ID is required";
    }

    return errors;
  };

  // Form submission handler
  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validate form
    const errors = validateForm();
    if (Object.keys(errors).length > 0) {
      setFormErrors(errors);
      return;
    }

    try {
      // Prepare data for API call
      const patientData = {
        ...formData,
        age: parseInt(formData.age),
        image: image, // Pass the actual File object
      };

      const addedPatient = await addPatient(patientData);
      if (addedPatient) {
        setSuccess(true);
        setAddedPatientId(addedPatient.id);

        // Reset form after 8 seconds
        setTimeout(() => {
          setFormData({
            name: "",
            age: "",
            id: "",
            notes: "",
          });
          setImage(null);
          setPreviewUrl(null);
          setSuccess(false);
          setAddedPatientId("");
        }, 8000);
      }
    } catch (err) {
      console.error("Error adding patient:", err);
    }
  };

  // Dismiss success popup
  const dismissSuccessPopup = () => {
    setSuccess(false);
    setAddedPatientId("");
    setFormData({
      name: "",
      age: "",
      id: "",
      notes: "",
    });
    setImage(null);
    setPreviewUrl(null);
  };

  // ===== STYLES =====

  // Page container style
  const pageStyle = {
    padding: isMobile ? "16px" : "32px",
    maxWidth: "1200px",
    margin: "0 auto",
    backgroundColor: "#FFF",
  };

  // Form container style
  const formContainerStyle = {
    backgroundColor: "#ffffff",
    borderRadius: "8px",
    boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1)",
    overflow: "hidden",
    border: "1px solid #f1f5f9",
  };

  // Form wrapper style
  const formWrapperStyle = {
    padding: isMobile ? "16px" : "32px",
  };

  // Two column layout
  const twoColumnLayoutStyle = {
    display: "grid",
    gridTemplateColumns: isMobile ? "1fr" : "1fr 1fr",
    gap: "30px",
  };

  // Section style
  const sectionStyle = {
    marginBottom: "30px",
  };

  // Section title
  const sectionTitleStyle = {
    fontSize: "18px",
    fontWeight: "600",
    color: "#4338ca", // Indigo color for section headers
    marginBottom: "20px",
    paddingBottom: "10px",
    borderBottom: "1px solid #e5e7eb",
  };

  // Form field style
  const formFieldStyle = {
    marginBottom: "20px",
  };

  // Label style
  const labelStyle = {
    display: "block",
    marginBottom: "6px",
    fontWeight: "500",
    fontSize: "15px",
    color: "#374151",
  };

  // Required asterisk style
  const requiredStyle = {
    color: "#ef4444",
    marginLeft: "4px",
  };

  // Input style
  const inputStyle = {
    width: "100%",
    padding: "10px 12px",
    fontSize: "15px",
    border: "1px solid #d1d5db",
    borderRadius: "6px",
    outline: "none",
    boxSizing: "border-box", // Ensures padding doesn't affect width
    transition: "border-color 0.2s ease",
  };

  // Textarea style
  const textareaStyle = {
    ...inputStyle,
    minHeight: "150px",
    resize: "vertical",
  };

  // Input focus style - applies on focus
  const focusStyle = (e) => {
    e.target.style.borderColor = "#3b82f6";
    e.target.style.boxShadow = "0 0 0 2px rgba(59, 130, 246, 0.1)";
  };

  // Input blur style - removes focus styles
  const blurStyle = (e) => {
    e.target.style.borderColor = "#d1d5db";
    e.target.style.boxShadow = "none";
  };

  // Error message style
  const errorMessageStyle = {
    color: "#ef4444",
    fontSize: "13px",
    marginTop: "4px",
    fontWeight: "500",
  };

  // Image upload area style
  const uploadAreaStyle = {
    border: "2px dashed #cbd5e1",
    borderRadius: "6px",
    backgroundColor: "#f8fafc",
    padding: "40px 20px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    textAlign: "center",
    cursor: "pointer",
    height: "400px",
    boxSizing: "border-box",
  };

  // Image icon style
  const imageIconStyle = {
    width: "48px",
    height: "48px",
    marginBottom: "20px",
    color: "#64748b",
  };

  // Select image button style
  const selectImageStyle = {
    display: "inline-block",
    color: "#4f46e5",
    fontSize: "16px",
    fontWeight: "500",
    marginBottom: "8px",
  };

  // Drag and drop text style
  const dragDropTextStyle = {
    color: "#6b7280",
    fontSize: "14px",
    marginBottom: "8px",
  };

  // Image format text
  const imageFormatTextStyle = {
    color: "#94a3b8",
    fontSize: "13px",
  };

  // Preview image container
  const previewImageContainerStyle = {
    position: "relative",
    height: "400px",
    width: "100%",
    borderRadius: "6px",
    overflow: "hidden",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  };

  // Preview image style
  const previewImageStyle = {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  };

  // Remove button container
  const removeButtonContainerStyle = {
    position: "absolute",
    top: "10px",
    right: "10px",
  };

  // Guidelines container style

  // Form footer style
  const formFooterStyle = {
    display: "flex",
    justifyContent: isMobile ? "center" : "flex-end",
    paddingTop: "24px",
    marginTop: "24px",
    borderTop: "1px solid #e5e7eb",
    flexDirection: isMobile ? "column" : "row",
    gap: "16px",
  };

  // Success popup overlay style
  const successOverlayStyle = {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    display: success ? "flex" : "none",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 1000,
    backdropFilter: "blur(4px)",
  };

  // Success popup container style
  const successPopupStyle = {
    backgroundColor: "white",
    borderRadius: "12px",
    boxShadow: "0 10px 25px rgba(0, 0, 0, 0.15)",
    padding: isMobile ? "24px 20px" : "32px",
    width: isMobile ? "90%" : "450px",
    maxWidth: "95%",
    position: "relative",
    animation: "popupFadeIn 0.3s ease-out",
    border: "1px solid #E2E8F0",
  };

  // Success popup header style
  const successHeaderStyle = {
    display: "flex",
    alignItems: "center",
    marginBottom: "24px",
  };

  // Success icon container style
  const successIconContainerStyle = {
    width: "48px",
    height: "48px",
    borderRadius: "50%",
    backgroundColor: "#DCFCE7",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    marginRight: "16px",
    flexShrink: 0,
  };

  // Success icon style
  const successIconStyle = {
    width: "28px",
    height: "28px",
    color: "#22C55E",
  };

  // Success title style
  const successTitleStyle = {
    fontSize: "20px",
    fontWeight: "600",
    color: "#111827",
  };

  // Success content style
  const successContentStyle = {
    padding: "16px 0",
    borderTop: "1px solid #E5E7EB",
    borderBottom: "1px solid #E5E7EB",
    marginBottom: "24px",
  };

  // Success message style
  const successMessageStyle = {
    fontSize: "16px",
    fontWeight: "500",
    color: "#4B5563",
    marginBottom: "16px",
  };

  // Patient ID section style
  const patientIdSectionStyle = {
    backgroundColor: "#F3F4F6",
    padding: "16px",
    borderRadius: "8px",
    marginTop: "16px",
  };

  // Patient ID label style
  const patientIdLabelStyle = {
    fontSize: "14px",
    color: "#6B7280",
    marginBottom: "8px",
  };

  // Patient ID display style
  const patientIdDisplayStyle = {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: "#FFFFFF",
    padding: "12px 16px",
    borderRadius: "6px",
    border: "1px solid #E5E7EB",
  };

  // Patient ID style
  const patientIdStyle = {
    fontSize: "18px",
    fontWeight: "600",
    color: "#2563EB",
    letterSpacing: "0.5px",
  };

  // Copy button style
  const copyButtonStyle = {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#EFF6FF",
    color: "#3B82F6",
    border: "1px solid #BFDBFE",
    borderRadius: "6px",
    padding: "6px 12px",
    fontSize: "14px",
    fontWeight: "500",
    cursor: "pointer",
    transition: "all 0.2s ease",
  };

  // Doctor info style
  const doctorInfoStyle = {
    fontSize: "14px",
    color: "#6B7280",
    marginTop: "12px",
    display: "flex",
    alignItems: "center",
  };

  // Info icon style
  const infoIconStyle = {
    width: "16px",
    height: "16px",
    color: "#9CA3AF",
    marginRight: "8px",
  };

  // Close button style
  const closeButtonStyle = {
    position: "absolute",
    top: "16px",
    right: "16px",
    backgroundColor: "transparent",
    border: "none",
    cursor: "pointer",
    padding: "4px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#9CA3AF",
  };

  // Close icon style
  const closeIconStyle = {
    width: "20px",
    height: "20px",
  };

  // Success buttons container style
  const successButtonsStyle = {
    display: "flex",
    justifyContent: "flex-end",
  };

  return (
    <Layout>
      <div style={pageStyle}>
        <h1
          style={{
            fontSize: "24px",
            fontWeight: "600",
            marginBottom: "24px",
            color: "#1e3a8a",
            borderLeft: "4px solid #4f46e5",
            paddingLeft: "12px",
          }}
        >
          Add New Patient
        </h1>

        {/* Success Popup */}
        <div style={successOverlayStyle}>
          <div style={successPopupStyle}>
            <button
              style={closeButtonStyle}
              onClick={dismissSuccessPopup}
              aria-label="Close"
            >
              <svg
                style={closeIconStyle}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>

            <div style={successHeaderStyle}>
              <div style={successIconContainerStyle}>
                <svg
                  style={successIconStyle}
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <h2 style={successTitleStyle}>Patient Added Successfully</h2>
            </div>

            <div style={successContentStyle}>
              <p style={successMessageStyle}>
                The patient has been successfully added to the system and can
                now be accessed by doctors.
              </p>

              <div style={patientIdSectionStyle}>
                <div style={patientIdLabelStyle}>Patient ID:</div>
                <div style={patientIdDisplayStyle}>
                  <span style={patientIdStyle}>{addedPatientId}</span>
                  <button
                    style={copyButtonStyle}
                    onClick={() => {
                      navigator.clipboard.writeText(addedPatientId);
                      alert("Patient ID copied to clipboard!");
                    }}
                  >
                    <svg
                      style={{
                        width: "16px",
                        height: "16px",
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
                        d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                    Copy ID
                  </button>
                </div>

                <div style={doctorInfoStyle}>
                  <svg
                    style={infoIconStyle}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  Doctors can search for this patient using this ID.
                </div>
              </div>
            </div>

            <div style={successButtonsStyle}>
              <Button
                variant="primary"
                onClick={dismissSuccessPopup}
                style={{
                  padding: "10px 16px",
                  fontSize: "14px",
                  borderRadius: "6px",
                }}
              >
                Add Another Patient
              </Button>
            </div>
          </div>
        </div>

        <div style={formContainerStyle}>
          <form onSubmit={handleSubmit}>
            <div style={formWrapperStyle}>
              <div style={twoColumnLayoutStyle}>
                {/* Personal Information Section */}
                <div style={sectionStyle}>
                  <h2 style={sectionTitleStyle}>Personal Information</h2>

                  <div style={formFieldStyle}>
                    <label style={labelStyle} htmlFor="name">
                      Patient Name
                      <span style={requiredStyle}>*</span>
                    </label>
                    <input
                      style={{
                        ...inputStyle,
                        borderColor: formErrors.name ? "#ef4444" : "#d1d5db",
                      }}
                      id="name"
                      name="name"
                      type="text"
                      value={formData.name}
                      onChange={handleInputChange}
                      placeholder="Enter full name"
                      onFocus={focusStyle}
                      onBlur={blurStyle}
                    />
                    {formErrors.name && (
                      <p style={errorMessageStyle}>{formErrors.name}</p>
                    )}
                  </div>

                  <div style={formFieldStyle}>
                    <label style={labelStyle} htmlFor="id">
                      Patient ID
                      <span style={requiredStyle}>*</span>
                    </label>
                    <input
                      style={{
                        ...inputStyle,
                        borderColor: formErrors.id ? "#ef4444" : "#d1d5db",
                      }}
                      id="id"
                      name="id"
                      type="text"
                      value={formData.id}
                      onChange={handleInputChange}
                      placeholder="Enter unique identifier (e.g., P123)"
                      onFocus={focusStyle}
                      onBlur={blurStyle}
                    />
                    {formErrors.id && (
                      <p style={errorMessageStyle}>{formErrors.id}</p>
                    )}
                  </div>

                  <div style={formFieldStyle}>
                    <label style={labelStyle} htmlFor="age">
                      Patient Age
                      <span style={requiredStyle}>*</span>
                    </label>
                    <input
                      style={{
                        ...inputStyle,
                        borderColor: formErrors.age ? "#ef4444" : "#d1d5db",
                      }}
                      id="age"
                      name="age"
                      type="number"
                      value={formData.age}
                      onChange={handleInputChange}
                      placeholder="Enter age in years"
                      onFocus={focusStyle}
                      onBlur={blurStyle}
                    />
                    {formErrors.age && (
                      <p style={errorMessageStyle}>{formErrors.age}</p>
                    )}
                  </div>

                  <div style={formFieldStyle}>
                    <label style={labelStyle} htmlFor="notes">
                      Notes (Optional)
                    </label>
                    <textarea
                      style={textareaStyle}
                      id="notes"
                      name="notes"
                      value={formData.notes}
                      onChange={handleInputChange}
                      placeholder="Enter any additional notes"
                      onFocus={focusStyle}
                      onBlur={blurStyle}
                    />
                  </div>
                </div>

                {/* Patient Image Section */}
                <div style={sectionStyle}>
                  <h2 style={sectionTitleStyle}>Patient Image</h2>

                  <div style={formFieldStyle}>
                    <label style={labelStyle}>Patient Image</label>

                    {previewUrl ? (
                      <div style={previewImageContainerStyle}>
                        <img
                          src={previewUrl}
                          alt="Patient"
                          style={previewImageStyle}
                        />
                        <div style={removeButtonContainerStyle}>
                          <Button
                            size="small"
                            variant="danger"
                            onClick={() => {
                              setImage(null);
                              setPreviewUrl(null);
                            }}
                          >
                            Remove
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <label htmlFor="image-upload" style={uploadAreaStyle}>
                        <svg
                          style={imageIconStyle}
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={1.5}
                            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                          />
                        </svg>
                        <span style={selectImageStyle}>Select image</span>
                        <p style={dragDropTextStyle}>or drag and drop</p>
                        <p style={imageFormatTextStyle}>
                           JPG JPEG up to 5MB
                        </p>
                        <input
                          id="image-upload"
                          name="image-upload"
                          type="file"
                          style={{ display: "none" }}
                          accept="image/*"
                          onChange={handleImageChange}
                        />
                      </label>
                    )}

                    {formErrors.image && (
                      <p style={errorMessageStyle}>{formErrors.image}</p>
                    )}
                  </div>
                </div>
              </div>

              <div style={formFooterStyle}>
                <Button
                  type="submit"
                  variant="primary"
                  disabled={loading}
                  style={{
                    width: isMobile ? "100%" : "auto",
                    minWidth: isMobile ? "auto" : "160px",
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
                      <span style={{ marginLeft: "8px" }}>Saving...</span>
                    </div>
                  ) : (
                    "Add Patient"
                  )}
                </Button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </Layout>
  );
};

export default AddPatient;
