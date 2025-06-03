from django.shortcuts import render
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import UserSerializer, CustomTokenObtainPairSerializer
from django.contrib.auth import get_user_model
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives import serialization
import hashlib
import json
import base64
from django.core.cache import cache
import time
import logging
from patients.utils import save_encryption_key

# Configure logger
logger = logging.getLogger(__name__)

User = get_user_model()

# Pre-generate DH parameters to improve performance
_dh_parameters = None

def get_dh_parameters():
    """
    Get cached DH parameters or generate new ones.
    Using 1024-bit keys instead of 2048-bit for better performance
    while maintaining adequate security.
    """
    global _dh_parameters
    if _dh_parameters is None:
        logger.info("Generating new DH parameters (one-time operation)")
        start_time = time.time()
        _dh_parameters = dh.generate_parameters(generator=2, key_size=1024)
        end_time = time.time()
        logger.info(f"DH parameters generated in {end_time - start_time:.2f} seconds")
    return _dh_parameters

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (permissions.AllowAny,)
    serializer_class = UserSerializer
    
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        
        return Response({
            "user": UserSerializer(user, context=self.get_serializer_context()).data,
            "message": "User registered successfully",
        }, status=status.HTTP_201_CREATED)

class DHKeyExchangeView(APIView):
    """
    API endpoint for Diffie-Hellman key exchange to establish secure encryption for image data
    
    GET: Initiates key exchange by generating server-side DH parameters and public key
    POST: Completes key exchange by receiving client's public key and computing shared secret
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """
        Generate DH parameters and server's key pair, then return public components
        """
        user_id = request.user.id
        logger.info(f"Initiating key exchange for user {user_id}")
        start_time = time.time()
        
        try:
            # Use pre-generated parameters for better performance
            parameters = get_dh_parameters()
            
            # Generate server's private key
            server_private_key = parameters.generate_private_key()
            
            # Get parameter values for serialization
            param_numbers = parameters.parameter_numbers()
            p = param_numbers.p
            g = param_numbers.g
            
            # Get server's private value for storage
            private_value = server_private_key.private_numbers().x
            
            # Get server's public key value
            public_numbers = server_private_key.public_key().public_numbers()
            public_value = public_numbers.y
            
            # Store server's private components in cache
            # Using a unique session key per user
            cache_key = f"dh_exchange_{user_id}"
            cache_data = {
                'private_value': str(private_value),
                'p': str(p),
                'g': str(g)
            }
            
            # Store with 30 minute expiration
            cache.set(cache_key, json.dumps(cache_data), 60 * 30)
            
            end_time = time.time()
            logger.info(f"Key exchange initialization completed in {end_time - start_time:.2f} seconds")
            
            # Return the DH parameters and server's public key
            return Response({
                'params': {
                    'p': str(p),
                    'g': str(g)
                },
                'server_public_key': str(public_value)
            })
        except Exception as e:
            logger.error(f"Error in key exchange initialization: {str(e)}")
            return Response({"error": f"Key exchange initialization failed: {str(e)}"}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def post(self, request):
        """
        Complete key exchange by receiving client's public key and computing shared secret
        """
        user_id = request.user.id
        logger.info(f"Completing key exchange for user {user_id}")
        start_time = time.time()
        
        # Get client's public key from request
        client_public_value = request.data.get('client_public_key')
        if not client_public_value:
            return Response({"error": "Missing client_public_key"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Retrieve server's stored DH components
        cache_key = f"dh_exchange_{user_id}"
        cached_data = cache.get(cache_key)
        
        if not cached_data:
            return Response(
                {"error": "Session expired or key exchange not initialized"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        cache_data = json.loads(cached_data)
        
        # Reconstruct DH objects
        p = int(cache_data['p'])
        g = int(cache_data['g'])
        private_value = int(cache_data['private_value'])
        
        try:
            # Convert client's public key to integer
            y = int(client_public_value)
            
            # Recreate parameter numbers and parameters
            param_numbers = dh.DHParameterNumbers(p, g)
            parameters = param_numbers.parameters()
            
            # Recreate server's private key
            priv_numbers = dh.DHPrivateNumbers(private_value, dh.DHPublicNumbers(y=y, parameter_numbers=param_numbers))
            server_private_key = priv_numbers.private_key()
            
            # Create client's public key
            public_numbers = dh.DHPublicNumbers(y=y, parameter_numbers=param_numbers)
            client_public_key = public_numbers.public_key()
            
            # Compute shared secret
            shared_secret = server_private_key.exchange(client_public_key)
            
            # Derive AES key using SHA-256
            aes_key = hashlib.sha256(shared_secret).digest()
            
            # Encode as base64 for storage
            key_b64 = base64.b64encode(aes_key).decode('utf-8')
            
            # Store AES key in cache keyed to the user
            encryption_key = f"encryption_key_{user_id}"
            cache.set(encryption_key, key_b64, 60 * 60 * 24)  # 24 hour expiration
            
            # Also save to persistent storage
            save_encryption_key(user_id, key_b64)
            
            end_time = time.time()
            logger.info(f"Key exchange completion finished in {end_time - start_time:.2f} seconds")
            
            # Success response
            return Response({'status': 'Key exchange successful'})
            
        except Exception as e:
            logger.error(f"Error completing key exchange: {str(e)}")
            return Response({"error": f"Key exchange failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer
