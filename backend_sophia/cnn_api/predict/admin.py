from django.contrib import admin
from .models import UserFeedback


@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ['id', 'model_prediction', 'user_prediction', 'created_at']
    list_filter = ['created_at', 'model_prediction', 'user_prediction']
    search_fields = ['model_prediction', 'user_prediction']
    readonly_fields = ['created_at', 'image_data']
    
    fieldsets = (
        ('Predictions', {
            'fields': ('model_prediction', 'user_prediction')
        }),
        ('Image Data', {
            'fields': ('image_data',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at',)
        }),
    )
from .models import Feedback

admin.site.register(Feedback)
