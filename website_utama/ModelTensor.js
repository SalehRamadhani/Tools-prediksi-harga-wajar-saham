
// Bagian 1 : Load Data dengan Papa Parse
let trainingData = [];
let testingData = [];
let inputMin, inputMax, labelMin, labelMax;

// Fungsi papa.parse untuk memproses file CSV
async function loadData(file, isTest = false) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,

      // Mengatur format koma dari CSV sehingga dapat diproses untuk kedepannya
      transform: function(value, column) {
        if (['EPS', 'PER', 'PBV', 'Harga'].includes(column)) {
          let parsedValue = parseFloat(value.replace(/\./g, '').replace(/,/g, '.'));
          return isNaN(parsedValue) ? null : parsedValue; // Mengembalikan null jika parsing gagal
        }
        return value;
      },
      complete: results => {
        if (isTest) {
          testingData = results.data.filter(d => d.EPS !== null && d.PER !== null && d.PBV !== null && d.Harga !== null);
        } else {
          trainingData = trainingData.concat(results.data.filter(d => d.EPS !== null && d.PER !== null && d.PBV !== null && d.Harga !== null));
        }
        resolve();
      },
      error: error => reject(error)
    });
  });
}

// Bagian 2 : Memisahkan data untuk data training dan data testing 
// Menghilangkan outlier
function cleanData(data) {
  return data.filter(d => {
    const valid = !isNaN(d.EPS) && !isNaN(d.PER) && !isNaN(d.PBV) && !isNaN(d.Harga);
    if (!valid) return false;

    // Menghitung mean dan standard deviation untuk setiap kolom
    const columns = ['EPS', 'PER', 'PBV', 'Harga'];
    const means = columns.map(col => data.reduce((sum, row) => sum + row[col], 0) / data.length);
    const stdDevs = columns.map((col, i) => Math.sqrt(data.map(row => Math.pow(row[col] - means[i], 2)).reduce((sum, val) => sum + val, 0) / data.length));

    // Menghitung batas atas dan bawah
    const lowerBounds = means.map((mean, i) => mean - 2 * stdDevs[i]);
    const upperBounds = means.map((mean, i) => mean + 2 * stdDevs[i]);

    // Memastikan data berada dalam batas yang ditentukan
    return columns.every((col, i) => d[col] >= lowerBounds[i] && d[col] <= upperBounds[i]);
  });
}

async function loadAndPrepareData() {
  const trainingYears = [2010, 2011, 2012, 2013, 2014, 2015,2016, 2017, 2018, 2019, 2020, 2021];
  const testingYear = 2022;

  try {
    const loadPromises = trainingYears.map(year => loadData(`/data/${year}.csv`));
    loadPromises.push(loadData(`/data/${testingYear}.csv`, true));
    await Promise.all(loadPromises);

    if (trainingData.length === 0 || testingData.length === 0) {
      console.error("Tidak ada data training atau data testing");
      return;
    }

    trainingData = cleanData(trainingData);
    testingData = cleanData(testingData);

  } catch (error) {
    console.error("Gagal Memuat Data:", error);
  }
}

// bagian 3 : mengubah data menjadi tensor
function convertToTensor(data) {
  return tf.tidy(() => {
    const inputs = data.map(d => [
      parseFloat(d.EPS),
      parseFloat(d.PER),
      parseFloat(d.PBV)
    ]);

    const labels = data.map(d => parseFloat(d.Harga));
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 3]);
    const labelTensor = tf.tensor1d(labels);

    const epsilon = 1e-7;
    const inputMax = inputTensor.max(0);
    const inputMin = inputTensor.min(0);
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin).add(epsilon));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin).add(epsilon));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

// Memproses dan mengonversi data
async function processAndConvertData() {
  await loadAndPrepareData();
  if (trainingData.length === 0 || testingData.length === 0) {
    return;
  }

  const tensorTrainingData = convertToTensor(trainingData);
  inputMin = tensorTrainingData.inputMin;
  inputMax = tensorTrainingData.inputMax;
  labelMin = tensorTrainingData.labelMin;
  labelMax = tensorTrainingData.labelMax;
}
processAndConvertData();

// bagian 4 : Merancang model
function createModel() {
  const model = tf.sequential();
  
  model.add(tf.layers.dense({
    inputShape: [3],
    units: 90,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
  }));
  
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.dense({
    units: 50,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
  }));
  
  model.add(tf.layers.dense({ units: 1 }));
  
  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError',
    metrics: ['mse', 'mae']
  });
  
  return model;
}

const model = createModel();
// menyimpan model
async function saveModel(model) {
  try {
    // Menyimpan model ke IndexedDB
    await model.save('indexeddb://my-model');
  } catch (error) {
    console.error('Gagal menyimpan model:', error);
  }
}

// bagian 5 : Melatih model
async function trainModel(model, trainingData, testingData) {
  const { inputs: trainInputs, labels: trainLabels } = trainingData;
  const { inputs: testInputs, labels: testLabels } = testingData;

  return await model.fit(trainInputs, trainLabels, {
    epochs: 1000,
    validationData: [testInputs, testLabels],
    shuffle: true,
    verbose: 1,
    batchSize: 16, 
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 20 }),
      new tf.CustomCallback({
        onEpochEnd: async (epoch, logs) => {
          if ((epoch + 1) % 10 === 0) {
            await saveModel(model);
            console.log(`Checkpoint disimpan untuk epoch ${epoch + 1}`);
          }
        }
      })
    ]
  });
}

// eksekusi latihan
async function runTraining() {
  await processAndConvertData();
  if (trainingData.length === 0 || testingData.length === 0) {
    console.error("training data atau testing data tidak tersedia");
    return;
  }
  const tensorTrainingData = convertToTensor(trainingData);
  const tensorTestingData = convertToTensor(testingData);

  // Validasi data tensor untuk NaN
  tensorTrainingData.inputs.data().then(data => {
    if (data.includes(NaN)) {
      console.error("Training inputs contain NaN");
    }
  });
 
  tensorTestingData.inputs.data().then(data => {
    if (data.includes(NaN)) {
      console.error("Testing inputs contain NaN");
    }
  });

  const history = await trainModel(model, tensorTrainingData, tensorTestingData);
  console.log(history.history);
}
runTraining();

predictFromLocalStorage();

// bagian 6 : Evaluasi model
async function evaluateModel(model, testingData) {
  const {inputs: testInputs, labels: testLabels} = testingData;

  try {
    const evalResult = await model.evaluate(testInputs, testLabels);
    console.log(`Hasil evaluasi - Loss: ${evalResult[0].dataSync()[0]}, 
    MSE: ${evalResult[1].dataSync()[0]}, 
    MAE: ${evalResult[2].dataSync()[0]}`);

    // Menyimpan nilai MAE untuk ditampilkan
    const mae = evalResult[2].dataSync()[0];
    console.log(`Mean Absolute Error (MAE): ${mae}`);

    // Melakukan prediksi
    const predictions = model.predict(testInputs);
    const predictedLabels = predictions.dataSync();
    const trueLabels = testLabels.dataSync();

    displayEvaluationResults(predictedLabels, trueLabels);
    const rmse = calculateRMSE(predictedLabels, trueLabels);
    console.log(`Root Mean Squared Error (RMSE): ${rmse}`);

    console.log('Hasil evaluasi:', evalResult);
  } catch (error) {
    console.error("Error during model evaluation:", error);
  }
}

function calculateRMSE(predictions, trueLabels) {
  let squaredErrorSum = 0;
  for (let i = 0; i < predictions.length; i++) {
    const error = predictions[i] - trueLabels[i];
    squaredErrorSum += Math.pow(error, 2);
  }
  const meanSquaredError = squaredErrorSum / predictions.length;
  const rmse = Math.sqrt(meanSquaredError);
  return rmse;
}

function displayEvaluationResults(predictedLabels, trueLabels) {
  console.log("Rangkuman Prediksi:");
  const sampleSize = 1; 
  for (let i = 0; i < sampleSize; i++) {
    console.log(`Sample ${i + 1}: Predicted = ${predictedLabels[i]}, Actual = ${trueLabels[i]}`);
  }
}

// menyiap fungsi dan model evaluasi
async function runEvaluation() {
  await loadAndPrepareData();
  if (!testingData || testingData.length === 0) {
    console.error("Data testing tidak tersedia atau tidak berhasil dimuat");
    return;
  }
  try {
    const model = await loadModel();
    if (!model) {
      console.error("Model tidak berhasil dimuat");
      return;
    }
    const tensorTestingData = convertToTensor(testingData);
    evaluateModel(model, tensorTestingData);
  } catch (error) {
    console.error("Terjadi kesalahan selama evaluasi model:", error);
  }
}
runEvaluation();

async function loadModel() {
  try {
    const model = await tf.loadLayersModel('indexeddb://my-model');
    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'meanSquaredError',
      metrics: ['mse', 'mae']
    });
    return model;
  } catch (error) {
    console.error('Gagal memuat model:', error);
    return null;
  }
}

// analisis sensitivitas
function perturbInput(input, perturbation) {
  return input.add(tf.randomUniform(input.shape, -perturbation, perturbation));
}

async function sensitivityAnalysis(model, testInputs, perturbationLevel) {
  const perturbedInput = perturbInput(testInputs, perturbationLevel);
  const originalOutput = model.predict(testInputs);
  const perturbedOutput = model.predict(perturbedInput);

  const changeInOutput = perturbedOutput.sub(originalOutput).abs();
  const sensitivity = changeInOutput.mean().dataSync();
  
  console.log(`Sensitivitas model pada tingkat perturbasi ${perturbationLevel}:`, sensitivity);
}

async function runSensitivityAnalysis() {
  await loadAndPrepareData();
  if (!testingData || testingData.length === 0) {
    console.error("Data testing tidak tersedia atau tidak berhasil dimuat");
    return;
  }
  try {
    const model = await loadModel();
    if (!model) {
      console.error("Model tidak berhasil dimuat");
      return;
    }
    const tensorTestingData = convertToTensor(testingData);
    const testInputs = tensorTestingData.inputs;

    const perturbationLevels = [0.01, 0.05, 0.1];
    for (const level of perturbationLevels) {
      await sensitivityAnalysis(model, testInputs, level);
    }
  } catch (error) {
    console.error("Terjadi kesalahan selama analisis sensitivitas:", error);
  }
}
runSensitivityAnalysis();

// Panggil predictFromLocalStorage() saat halaman dimuat
document.addEventListener('DOMContentLoaded', predictFromLocalStorage);
async function predictFromLocalStorage() {
  await processAndConvertData(); // Menunggu proses pemrosesan selesai
  const eps = parseFloat(localStorage.getItem("EPS"));
  const per = parseFloat(localStorage.getItem("PER"));
  const pbv = parseFloat(localStorage.getItem("PBV"));

  if (isNaN(eps) || isNaN(per) || isNaN(pbv)) {
    alert('Harap masukkan nilai numerik yang valid.');
    localStorage.removeItem("fairValue");
    return;
  }

  // Normalisasi input sesuai dengan data training
  const normalizedInput = normalizeInput([eps, per, pbv]);

  const model = await loadModel();
  if (!model) {
    console.error("Model tidak berhasil dimuat");
    return;
  }

  // Prediksi harga saham
  const inputTensor = tf.tensor2d([normalizedInput], [1, 3]);
  const prediction = model.predict(inputTensor);
  const predictedPrice = denormalizeLabel(prediction.dataSync()[0]);

  localStorage.setItem('fairValue', predictedPrice.toFixed(2));

  updateRecommended();
  updateDecision();
}

async function predictPrice(model, normalizedInput) {
  const inputTensor = tf.tensor2d([normalizedInput], [1, 3]);
  const prediction = model.predict(inputTensor);
  const predictedPrice = denormalizeLabel(prediction.dataSync()[0]);
  return predictedPrice;
}

function normalizeInput(input) {
  const [eps, per, pbv] = input;
  const normalizedEps = (eps - inputMin.dataSync()[0]) / (inputMax.dataSync()[0] - inputMin.dataSync()[0]);
  const normalizedPer = (per - inputMin.dataSync()[1]) / (inputMax.dataSync()[1] - inputMin.dataSync()[1]);
  const normalizedPbv = (pbv - inputMin.dataSync()[2]) / (inputMax.dataSync()[2] - inputMin.dataSync()[2]);
  return [normalizedEps, normalizedPer, normalizedPbv];
}

function denormalizeLabel(label) {
  return label * (labelMax.dataSync()[0] - labelMin.dataSync()[0]) + labelMin.dataSync()[0];
}

async function loadModel() {
  try {
    const model = await tf.loadLayersModel('indexeddb://my-model');
    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'meanSquaredError',
      metrics: ['mse', 'mae']
    });
    return model;
  } catch (error) {
    console.error('Gagal memuat model:', error);
    return null;
  }
}
