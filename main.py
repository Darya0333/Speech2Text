import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal

import os
import queue
import re
import sys
import time

from google.cloud import speech
import pyaudio

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"./credential.json"

STREAMING_LIMIT = 240000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)

def get_current_time() -> int:
    return int(round(time.time() * 1000))

class ResumableMicrophoneStream:
    def __init__(
        self: object,
        rate: int,
        chunk_size: int,
    ) -> None:
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._fill_buffer,
        )

    def __enter__(self: object) -> object:
        self.closed = False
        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> object:
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses: object, stream: object) -> None:
    for response in responses:
        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (
            stream.result_end_time
            - stream.bridging_offset
            + (STREAMING_LIMIT * stream.restart_counter)
        )

        if result.is_final:
            print(str(corrected_time) + ": " + transcript + "\n")

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting...\n")
                stream.closed = True
                break
        else:
            print(str(corrected_time) + ": " + transcript + "\r")
            stream.last_transcript_was_final = False

class WorkerThread(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        try:
            client = speech.SpeechClient()
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                language_code="ja-JP",
                max_alternatives=1,
            )

            streaming_config = speech.StreamingRecognitionConfig(
                config=config, interim_results=True
            )

            mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
            print('Listening, say "Quit" or "Exit" to stop.')

            with mic_manager as stream:
                while not stream.closed:
                    print("\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n")

                    stream.audio_input = []
                    audio_generator = stream.generator()

                    requests = (
                        speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator
                    )

                    responses = client.streaming_recognize(streaming_config, requests)

                    listen_print_loop(responses, stream)

                    if stream.result_end_time > 0:
                        stream.final_request_end_time = stream.is_final_end_time
                    stream.result_end_time = 0
                    stream.last_audio_input = []
                    stream.last_audio_input = stream.audio_input
                    stream.audio_input = []
                    stream.restart_counter = stream.restart_counter + 1

                    if not stream.last_transcript_was_final:
                        print("\n")
                    stream.new_stream = True
        except Exception as e:
            print(f"An unknown error occurred: {type(e).__name__} - {str(e)}")

class QRecordButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: #fff;
                padding: 10px 20px;
                border-radius: 10px;
            }
            QPushButton:pressed {
                background-color: #c0392b;
            }
        """)
        self.setText("Record Audio")
        self.setCheckable(True)
        self.toggled.connect(self.start_stop_recording)

    def start_stop_recording(self, checked):
        if checked:
            self.worker = WorkerThread()
            self.worker.progress.connect(self.on_progress)
            self.worker.start()
            self.setText("Stop Recording")
        else:
            self.worker.terminate()
            self.setText("Record Audio")

    def on_progress(self, progress):
        print(f"Progress: {progress}%")

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech recognition")
        self.setGeometry(100, 100, 320, 120)

        layout = QVBoxLayout()

        label = QLabel("Speech recognition")
        record_button = QRecordButton()
        layout.addWidget(label)
        layout.addWidget(record_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #333;
            color: #fff;
        }
        QLabel {
            color: #fff;
            padding-left: 8px;  
        }
    """)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())