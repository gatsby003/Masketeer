# **Masketeer**

Aadhar masking implementation using Pytesseract, OpenCV, and FastAPI.

![https://www.worldhistory.org/img/r/p/500x600/15130.png?v=1642146798](https://www.worldhistory.org/img/r/p/500x600/15130.png?v=1642146798)

### Local Setup Instructions

1. Setup virtual environment and install requirements
2. Start fast api server using 

```yaml
uvicorn main:app --reload
```

1. Visit [localhost:8000/docs](http://localhost:8000/docs)  for endpoint documentation 

## Todo

- [ ]  Handle edge cases and low-res images
- [ ]  Add more features such compression , alignment and cropping
- [ ]  bacup data to a database so clients can access it later
- [ ]  implement batch processing for large number of cards
- [ ]  deploy to cloud
