from django.contrib import admin
from django import forms
# Register your models here.
from .models import User

class UserAdminForm(forms.ModelForm):
    class Meta:
        model = User
        fields = '__all__'
    def clean(self):
        cleaned_data = super().clean()
        
class UserAdmin(admin.ModelAdmin):
    form = UserAdminForm
    list_display = ('username', 'email')
    search_fields = ('username',)

admin.site.register(User,UserAdmin)