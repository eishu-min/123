let tfliteModel, history = [];
const issues = ['Acne','Dark Spots','Redness','Wrinkles'];

async function initModel() {
  await tf.setBackend('wasm');
  const tflite = await tfliteTFLite.loadTFLiteModel({
    modelUrl: 'model/skin_model_web.tflite',
    inputShape: [224,224,3]
  });
  tfliteModel = tflite;
}

function startCamera() {
  const video = document.getElementById('video');
  navigator.mediaDevices.getUserMedia({ video:true })
    .then(stream => video.srcObject = stream);
}

function captureAndAnalyze(img) {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = 224; canvas.height = 224;
  ctx.drawImage(img, 0,0,224,224);
  const input = tf.browser.fromPixels(canvas)
    .expandDims(0).toFloat().div(255);
  
  const output = tfliteModel.predict(input);
  const scores = Array.from(output.dataSync());
  showResults(scores);
  history.push({ time: new Date(), scores });
  showHistory();
}

function showResults(scores) {
  const div = document.getElementById('results');
  div.innerHTML = '';
  scores.forEach((s,i) => {
    const p = document.createElement('p');
    p.className = 'result-item';
    p.textContent = `${issues[i]}: ${(s*100).toFixed(1)}%`;
    div.appendChild(p);
  });
}

function showHistory() {
  const ul = document.getElementById('history');
  ul.innerHTML = '';
  history.forEach(h => {
    const li = document.createElement('li');
    const t = new Date(h.time).toLocaleString();
    const scores = h.scores.map((s,i)=>`${issues[i]} ${(s*100).toFixed(1)}%`).join('; ');
    li.textContent = `${t} → ${scores}`;
    ul.appendChild(li);
  });
}

window.addEventListener('DOMContentLoaded', async () => {
  await initModel();
  document.getElementById('start-camera').onclick = startCamera;
  document.getElementById('capture').onclick = () => {
    const video = document.getElementById('video');
    captureAndAnalyze(video);
  };
  document.getElementById('upload').onchange = e => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => captureAndAnalyze(img);
    img.src = URL.createObjectURL(file);
  };
});
