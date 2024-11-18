# Real Time Speech To Text (Using OpenAi Whisper)

## Requirements
- Python 3.1x

## Setting Up
- Install requirements using `pip install -r requirements.txt`

## Running the Application
- Run the flask app using `python3 app.py`

## Selecting the Appropriate Model
Whisper offers several models that balance speed and accuracy:

- `tiny`: Fastest but least accurate
- `base`: A balance between speed and accuracy
- `small`: More accurate, slower than base
- `medium`: Even more accurate, slower than small
- `large`: Most accurate but slowest

You can select a model by specifying it when loading the Whisper model. For example:
```python
self.model = whisper.load_model("medium")
```


## Demo
![sample](https://github.com/user-attachments/assets/90d45012-5f1d-4fc1-b72d-5a47c3eb4c63)

## To-Dos
- [ ] Improve accuracy of transcription
- [ ] Add support for multiple languages
- [ ] Optimize performance for low-latency environments
- [ ] Implement speaker recognition
- [ ] Webohook - Create separate sessions(?) for each connected client

## Contribution Guidelines
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

## License
This project is licensed under the MIT License.
