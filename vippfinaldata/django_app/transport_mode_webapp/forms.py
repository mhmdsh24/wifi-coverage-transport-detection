from django import forms

class UploadFileForm(forms.Form):
    """Form for uploading files for transport mode detection"""
    file = forms.FileField(
        label='Select a file',
        help_text='Supported formats: CSV, JSON',
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )
    
    def clean_file(self):
        file = self.cleaned_data.get('file')
        
        if file:
            # Check file extension
            if not (file.name.endswith('.csv') or file.name.endswith('.json')):
                raise forms.ValidationError('Unsupported file format. Please upload a CSV or JSON file.')
            
            # Check file size (max 10MB)
            if file.size > 10 * 1024 * 1024:
                raise forms.ValidationError('File size too large. Maximum file size is 10MB.')
                
        return file 