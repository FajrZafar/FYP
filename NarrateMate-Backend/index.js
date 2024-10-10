
// trying for microphone- speech- to- text...final code with microphone integrated
const express = require('express');
const multer = require('multer');
const path = require('path');
const { spawn, exec } = require('child_process'); 
const WebSocket = require('ws');
const fs = require('fs');
const pdfkit = require('pdfkit');
const app = express();
const port = 3000;


app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configure multer storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});

const upload = multer({ storage });

app.get('/', (req, res) => {
  res.send('Hello World');
});

// Handle image upload and processing
app.post('/upload', upload.single('image'), (req, res) => {
  const filePath = path.join(__dirname, 'uploads', req.file.filename);
  const resultsDir = path.join(__dirname, 'results');
  const folderName = path.basename(req.file.filename, path.extname(req.file.filename));
  const extractedTextFilePath = path.join(resultsDir, folderName, 'extracted_text.txt');
  const pdfFilePath = path.join(resultsDir, folderName, 'extracted_text.pdf');
  const imageName = path.basename(filePath, path.extname(filePath));

  console.log(`File path: ${filePath}`);
  console.log(`Results directory: ${resultsDir}`);
  console.log(`Expected extracted text file path: ${extractedTextFilePath}`);
  console.log(`PDF file path: ${pdfFilePath}`);
  console.log(`Image name: ${imageName}`);

  if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir, { recursive: true });
  }

  exec(`python3 extract_text.py "${filePath}" "${resultsDir}"`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error: ${stderr}`);
      return res.status(500).json({ error: stderr });
    }

    console.log(`Python stdout: ${stdout}`);

    fs.readFile(extractedTextFilePath, 'utf8', (err, extractedText) => {
      if (err) {
        console.error(`Failed to read extracted text: ${err.message}`);
        return res.status(500).json({ error: 'Failed to read extracted text' });
      }

      console.log(`Extracted text read from file: ${extractedText}`);

      const doc = new pdfkit();
      doc.pipe(fs.createWriteStream(pdfFilePath));
      doc.text(extractedText);
      doc.end();

      res.json({
        message: 'Image processed successfully',
        extractedText: extractedText.trim(),
        pdfFilePath: pdfFilePath
      });

      // const captionProcess = spawn('python3', ['generate_captions.py', filePath, resultsDir, imageName]);

      // captionProcess.stdout.on('data', (data) => {
      //   console.log(`Python stdout (captions): ${data}`);
      // });

      // captionProcess.stderr.on('data', (data) => {
      //   console.error(`Python stderr (captions): ${data}`);
      // });

      // captionProcess.on('close', (captionCode) => {
      //   if (captionCode === 0) {
      //     const captionFilePath = path.join(resultsDir, 'captions.txt');
      //     const captions = fs.readFileSync(captionFilePath, 'utf8');
      //     broadcastCaptions(captions);
      //   } else {
      //     console.error('Caption generation failed.');
      //   }
      //});
    });
  });
});

// Question Answering API
app.post('/api/qa', (req, res) => {
  const { question, pdfFilePath } = req.body;

  console.log(`Received question: ${question}`);
  console.log(`PDF File path: ${pdfFilePath}`);

  const qaProcess = spawn('python3', ['question_answering.py', pdfFilePath, question]);

  let qaOutput = '';

  qaProcess.stdout.on('data', (data) => {
    console.log(`Python stdout (QA): ${data}`);
    qaOutput += data;
  });

  qaProcess.stderr.on('data', (data) => {
    console.error(`Python stderr (QA): ${data}`);
  });

  qaProcess.on('close', (qaCode) => {
    if (qaCode === 0) {
      console.log(`Question answered successfully: ${qaOutput}`);
      res.json({
        message: 'Question answered successfully',
        answer: qaOutput.trim(),
      });
    } else {
      console.error('Question answering failed.');
      res.status(500).json({ error: 'Question answering failed' });
    }
  });
});

// Handle audio file upload and transcription
app.post('/api/voice-to-text', upload.single('audio'), (req, res) => {
  const audioFilePath = path.join(__dirname, 'uploads', req.file.filename);

  console.log(`Audio file path: ${audioFilePath}`);

  // Run the transcription script
  const transcriptionProcess = spawn('python3', ['transcribe_audio.py', audioFilePath]);

  let transcriptionOutput = '';

  transcriptionProcess.stdout.on('data', (data) => {
    console.log(`Python stdout (transcription): ${data}`);
    transcriptionOutput += data;
  });

  transcriptionProcess.stderr.on('data', (data) => {
    console.error(`Python stderr (transcription): ${data}`);
  });

  transcriptionProcess.on('close', (transcriptionCode) => {
    if (transcriptionCode === 0) {
      console.log(`Audio transcription successful: ${transcriptionOutput}`);
      res.json({
        transcription: transcriptionOutput.trim(),
      });
    } else {
      console.error('Audio transcription failed.');
      res.status(500).json({ error: 'Audio transcription failed' });
    }
  });
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
