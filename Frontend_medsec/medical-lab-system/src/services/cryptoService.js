import * as forge from "node-forge";

/**
 * Initialize Diffie-Hellman key exchange with server
 * @param {string} accessToken - JWT access token
 * @returns {Promise<Object>} - DH parameters and server public key
 */
export async function initializeKeyExchange(accessToken) {
  const response = await fetch("http://localhost:8000/api/auth/key-exchange/", {
    method: "GET",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Key exchange initialization failed: ${response.status}`);
  }

  return await response.json();
}

/**
 * Convert a string to BigInteger
 * @param {string} str - Numeric string to convert
 * @returns {forge.jsbn.BigInteger} - BigInteger object
 */
function stringToBigInt(str) {
  return new forge.jsbn.BigInteger(str);
}

/**
 * Generate a cryptographically secure random BigInteger
 * @param {forge.jsbn.BigInteger} max - Upper bound (exclusive)
 * @returns {forge.jsbn.BigInteger} - Random BigInteger
 */
function generateRandomBigInt(max) {
  // Create a byte array of appropriate length
  const bitLength = max.bitLength();
  const byteLength = Math.ceil(bitLength / 8);

  // Generate random bytes
  const randomBytes = forge.random.getBytesSync(byteLength);

  // Convert to BigInteger and ensure it's less than max
  const randomBigInt = new forge.jsbn.BigInteger(
    forge.util.bytesToHex(randomBytes),
    16
  );

  // Modulo to ensure value is in valid range
  return randomBigInt.mod(max);
}

/**
 * Complete the Diffie-Hellman key exchange
 * @param {string} accessToken - JWT access token
 * @param {Object} dhParams - Parameters from initializeKeyExchange
 * @returns {Promise<string>} - Status message
 */
export async function completeKeyExchange(accessToken, dhParams) {
  try {
    // Parse parameters from server
    const p = stringToBigInt(dhParams.params.p);
    const g = stringToBigInt(dhParams.params.g);
    const serverPublicKey = stringToBigInt(dhParams.server_public_key);

    // Generate client's private key (keep this secret)
    const clientPrivateKey = generateRandomBigInt(p);

    // Calculate client's public key: g^privateKey mod p
    const clientPublicKey = g.modPow(clientPrivateKey, p);

    // Send client's public key to server
    const response = await fetch(
      "http://localhost:8000/api/auth/key-exchange/",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          client_public_key: clientPublicKey.toString(),
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`Key exchange completion failed: ${response.status}`);
    }

    // Calculate shared secret (this is never transmitted)
    const sharedSecret = serverPublicKey.modPow(clientPrivateKey, p);

    // Derive AES key from shared secret using SHA-256
    const md = forge.md.sha256.create();
    md.update(sharedSecret.toString());
    const aesKey = md.digest().toHex();

    // Store the derived key securely (in memory/session storage, not localStorage)
    sessionStorage.setItem("encryption_established", "true");

    console.log("Key exchange completed successfully");
    return await response.json();
  } catch (error) {
    console.error("Key exchange failed:", error);
    throw error;
  }
}

/**
 * Refresh the encryption key
 * @param {string} accessToken - JWT access token
 * @returns {Promise<boolean>} - Whether refresh was successful
 */
export async function refreshEncryptionKey(accessToken) {
  try {
    const dhParams = await initializeKeyExchange(accessToken);
    await completeKeyExchange(accessToken, dhParams);
    console.log("Encryption key refreshed successfully");
    return true;
  } catch (error) {
    console.error("Failed to refresh encryption key:", error);
    return false;
  }
}

/**
 * Set up periodic key refresh
 * @param {number} intervalMinutes - Minutes between refreshes
 * @returns {Function} - Cleanup function
 */
export function setupKeyRefresh(intervalMinutes = 60) {
  // Clear any existing refresh intervals
  if (window.keyRefreshInterval) {
    clearInterval(window.keyRefreshInterval);
  }

  // Set up new refresh interval
  window.keyRefreshInterval = setInterval(async () => {
    const accessToken = localStorage.getItem("accessToken");

    if (!accessToken) {
      clearInterval(window.keyRefreshInterval);
      return;
    }

    await refreshEncryptionKey(accessToken);
  }, intervalMinutes * 60 * 1000);

  // Return cleanup function
  return () => {
    if (window.keyRefreshInterval) {
      clearInterval(window.keyRefreshInterval);
    }
  };
}
