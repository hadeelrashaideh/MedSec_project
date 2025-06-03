from rest_framework import permissions

class IsDoctorUser(permissions.BasePermission):
    """
    Custom permission to only allow doctors to access certain views.
    """
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == 'DOCTOR'

class IsLabUser(permissions.BasePermission):
    """
    Custom permission to only allow lab personnel to access certain views.
    """
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == 'LAB' 