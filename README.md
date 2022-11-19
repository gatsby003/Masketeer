# Highly Available Aadhaar Masking Service

# **Masketeer**

Aadhar masking implementation using Pytesseract, OpenCV, and FastAPI.

### Local Setup Instructions

1. Setup virtual environment and install requirements
2. Start fast api server using 

```yaml
uvicorn main:app --reload
```

1. Visit [localhost:8000/docs](http://localhost:8000/docs)  for endpoint docuementation 

## Todo

- [ ]  Add more features such compression , alignment and cropping
- [ ]  bacup data to a database so clients can access it later
- [ ]  implement batch processing for large number of cards
- [ ]  deploy to cloud
