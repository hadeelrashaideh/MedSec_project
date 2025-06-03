from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from django.contrib.auth import get_user_model

User = get_user_model()

class AuthenticationTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.register_url = reverse('register')
        self.login_url = reverse('token_obtain_pair')
        
        self.doctor_data = {
            'email': 'doctor@example.com',
            'password': 'testpassword123',
            'first_name': 'John',
            'last_name': 'Doe',
            'role': 'DOCTOR'
        }
        
        self.lab_data = {
            'email': 'lab@example.com',
            'password': 'testpassword123',
            'first_name': 'Lab',
            'last_name': 'Manager',
            'role': 'LAB'
        }

    def test_register_doctor(self):
        response = self.client.post(self.register_url, self.doctor_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(User.objects.count(), 1)
        self.assertEqual(User.objects.get().role, 'DOCTOR')
        
    def test_register_lab(self):
        response = self.client.post(self.register_url, self.lab_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(User.objects.count(), 1)
        self.assertEqual(User.objects.get().role, 'LAB')
        
    def test_login_doctor(self):
        # First create user
        self.client.post(self.register_url, self.doctor_data, format='json')
        
        # Attempt login
        login_data = {
            'email': 'doctor@example.com',
            'password': 'testpassword123'
        }
        response = self.client.post(self.login_url, login_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue('access' in response.data)
        self.assertTrue('refresh' in response.data)
        self.assertEqual(response.data['role'], 'DOCTOR')
        
    def test_login_lab(self):
        # First create user
        self.client.post(self.register_url, self.lab_data, format='json')
        
        # Attempt login
        login_data = {
            'email': 'lab@example.com',
            'password': 'testpassword123'
        }
        response = self.client.post(self.login_url, login_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue('access' in response.data)
        self.assertTrue('refresh' in response.data)
        self.assertEqual(response.data['role'], 'LAB')
