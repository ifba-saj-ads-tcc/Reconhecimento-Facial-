const abhay = []
const adil = []
const akshay = []
const amitabh = []
const amole = []
const angela = []
const anushka = []
const arshad = []
const ayushmann = []
const celine = []
const cher = []
const chiranjeevi = []
const chris = []
const hrithik = []
const jacinda = []
const johnny = []
const kalki = []
const kamala = []
const madonna = []
const meryl = []
const mohanlal = []
const nancy = []
const oprah = []
const prabhas = []
const prabhu = []
const rihanna = []
const salman = []
const scarlett = []
const sheryl = []
const whitney = []

const imageSize = 96
let neuralNetwork 

function failureCallback(e) {
  console.error(e)
}
function successCallback(img) {
  img.resize(imageSize, imageSize)
}

function preload() {
  for (let i = 0; i < 1; i++) {
    abhay[i] = loadImage(`./output/train/abhay_deol/foto (${i + 1}).jpg`, successCallback, failureCallback)
    adil[i] = loadImage(`./output/train/adil_hussain/foto (${i + 1}).jpg`, successCallback, failureCallback)
    akshay[i] = loadImage(`./output/train/akshay_kumar/foto (${i + 1}).jpg`, successCallback, failureCallback)
    amitabh[i] = loadImage(`./output/train/amitabh_bachchan/foto (${i + 1}).jpg`, successCallback, failureCallback)
    amole[i] = loadImage(`./output/train/amole_gupte/foto (${i + 1}).jpg`, successCallback, failureCallback)
    angela[i] = loadImage(`./output/train/Angela Merkel/foto (${i + 1}).jpg`, successCallback, failureCallback)
    anushka[i] = loadImage(`./output/train/anushka_shetty/foto (${i + 1}).jpg`, successCallback, failureCallback)
    arshad[i] = loadImage(`./output/train/arshad_warsi/foto (${i + 1}).jpg`, successCallback, failureCallback)
    ayushmann[i] = loadImage(`./output/train/ayushmann_khurrana/foto (${i + 1}).jpg`, successCallback, failureCallback)
    celine[i] = loadImage(`./output/train/Celine Dion/foto (${i + 1}).jpg`, successCallback, failureCallback)
    cher[i] = loadImage(`./output/train/Cher/foto (${i + 1}).jpg`, successCallback, failureCallback)
    chiranjeevi[i] = loadImage(`./output/train/chiranjeevi/foto (${i + 1}).jpg`, successCallback, failureCallback)
    chris[i] = loadImage(`./output/train/chris evans/foto (${i + 1}).jpg`, successCallback, failureCallback)
    hrithik[i] = loadImage(`./output/train/hrithik roshan/foto (${i + 1}).jpg`, successCallback, failureCallback)
    jacinda[i] = loadImage(`./output/train/Jacinda Ardern/foto (${i + 1}).jpg`, successCallback, failureCallback)
    johnny[i] = loadImage(`./output/train/johnny depp/foto (${i + 1}).jpg`, successCallback, failureCallback)
    kalki[i] = loadImage(`./output/train/kalki_koechlin/foto (${i + 1}).jpg`, successCallback, failureCallback)
    kamala[i] = loadImage(`./output/train/Kamala Harris/foto (${i + 1}).jpg`, successCallback, failureCallback)
    madonna[i] = loadImage(`./output/train/Madonna/foto (${i + 1}).jpg`, successCallback, failureCallback)
    meryl[i] = loadImage(`./output/train/Meryl Streep/foto (${i + 1}).jpg`, successCallback, failureCallback)
    mohanlal[i] = loadImage(`./output/train/mohanlal/foto (${i + 1}).jpg`, successCallback, failureCallback)
    nancy[i] = loadImage(`./output/train/Nancy Pelosi/foto (${i + 1}).jpg`, successCallback, failureCallback)
    oprah[i] = loadImage(`./output/train/Oprah Winfrey/foto (${i + 1}).jpg`, successCallback, failureCallback)
    prabhas[i] = loadImage(`./output/train/prabhas/foto (${i + 1}).jpg`, successCallback, failureCallback)
    prabhu[i] = loadImage(`./output/train/prabhu_deva/foto (${i + 1}).jpg`, successCallback, failureCallback)
    rihanna[i] = loadImage(`./output/train/Rihanna/foto (${i + 1}).jpg`, successCallback, failureCallback)
    salman[i] = loadImage(`./output/train/salman_khan/foto (${i + 1}).jpg`, successCallback, failureCallback)
    scarlett[i] = loadImage(`./output/train/scarlett johansson/foto (${i + 1}).jpg`, successCallback, failureCallback)
    sheryl[i] = loadImage(`./output/train/Sheryl Sandberg/foto (${i + 1}).jpg`, successCallback, failureCallback)
    whitney[i] = loadImage(`./output/train/Whitney Houston/foto (${i + 1}).jpg`, successCallback, failureCallback)
  }
}

const sequential1 = [
  {
    type: 'conv2d',
    filters: 64,
    padding: 'same',
    kernelSize: 3,
    strides: 1,
  },
  {
    type: 'batchNormalization'
  },
  {
    type: 'activation',
    activation: 'relu',
  },
  {
    type: 'conv2d',
    filters: 64,
    kernelSize: 3,
    padding: 'same'
  },
  {
    type: 'batchNormalization'
  },
  {
    type: 'activation',
    activation: 'relu',
  },
]
const sequential2 = [
  {
    type: 'conv2d',
    filters: 128,
    padding: 'same',
    kernelSize: 3,
    strides: 1,
  },
  {
    type: 'batchNormalization'
  },
  {
    type: 'activation',
    activation: 'relu',
  },
  {
    type: 'conv2d',
    filters: 128,
    kernelSize: 3,
    padding: 'same'
  },
  {
    type: 'batchNormalization'
  },
  {
    type: 'activation',
    activation: 'relu',
  },
]
const sequential3 = [
  {
    type: 'conv2d',
    filters: 256,
    padding: 'same',
    kernelSize: 3,
    strides: 1,
  },
  {
    type: 'batchNormalization'
  },
  {
    type: 'activation',
    activation: 'relu',
  },
  {
    type: 'conv2d',
    filters: 256,
    kernelSize: 3,
    padding: 'same'
  },
  {
    type: 'batchNormalization'
  },
  {
    type: 'activation',
    activation: 'relu',
  },
]
const sequential4 = [
  {
    type: 'conv2d',
    filters: 512,
    padding: 'same',
    kernelSize: 3,
    strides: 1,
  },
  {
    type: 'batchNormalization'
  },
  {
    type: 'activation',
    activation: 'relu',
  },
  {
    type: 'conv2d',
    filters: 512,
    kernelSize: 3,
    padding: 'same'
  },
  {
    type: 'batchNormalization'
  },
  {
    type: 'activation',
    activation: 'relu',
  },
]




const layers = [
  {
    type: 'conv2d',
    filters: 64,
    kernelSize: 7,
    strides: 2,
    padding: 'same'
  },
  {
    type: 'batchNormalization'
  },
  {
    type: 'activation',
    activation: 'relu',
  },
  {
    type: 'maxPooling2d',
    poolSize: 3,
    strides: 2,
    padding: 'same'
  },
  ...sequential1,
  ...sequential2,
  ...sequential3,
  ...sequential4,
  {
    type: 'globalAveragePooling2d',
  },
  {
    type: 'dense',
    kernelInitializer: 'varianceScaling',
    activation: 'softmax',
  },
];

function setup() {
  const options = {
    inputs: [imageSize, imageSize, 4],
    task: "imageClassification",
    layers: layers,
    learningRate: 0.2,
    debug: true,
  }

  neuralNetwork = ml5.neuralNetwork(options)

  for (let i = 0; i < angela.length; i++) {
    neuralNetwork.addData({ image: abhay[i] }, { label: "abhay" })
     neuralNetwork.addData({ image: adil[i] }, { label: "adil" })
    neuralNetwork.addData({ image: akshay[i] }, { label: "akshay" })
    neuralNetwork.addData({ image: amitabh[i] }, { label: "amitabh" })
    neuralNetwork.addData({ image: amole[i] }, { label: "amole" })
    neuralNetwork.addData({ image: angela[i] }, { label: "angela" })
    neuralNetwork.addData({ image: anushka[i] }, { label: "anushka" })
    neuralNetwork.addData({ image: arshad[i] }, { label: "arshad" })
    neuralNetwork.addData({ image: ayushmann[i] }, { label: "ayushmann" })
    neuralNetwork.addData({ image: celine[i] }, { label: "celine" })
    neuralNetwork.addData({ image: cher[i] }, { label: "cher" })
    neuralNetwork.addData({ image: chiranjeevi[i] }, { label: "chiranjeevi" })
    neuralNetwork.addData({ image: chris[i] }, { label: "chris" })
    neuralNetwork.addData({ image: hrithik[i] }, { label: "hrithik" })
    neuralNetwork.addData({ image: jacinda[i] }, { label: "jacinda" })
    neuralNetwork.addData({ image: johnny[i] }, { label: "johnny" })
    neuralNetwork.addData({ image: kalki[i] }, { label: "kalki" })
    neuralNetwork.addData({ image: kamala[i] }, { label: "kamala" })
    neuralNetwork.addData({ image: madonna[i] }, { label: "madonna" })
    neuralNetwork.addData({ image: meryl[i] }, { label: "meryl" })
    neuralNetwork.addData({ image: mohanlal[i] }, { label: "mohanlal" })
    neuralNetwork.addData({ image: nancy[i] }, { label: "nancy" })
    neuralNetwork.addData({ image: oprah[i] }, { label: "oprah" })
    neuralNetwork.addData({ image: prabhas[i] }, { label: "prabhas" })
    neuralNetwork.addData({ image: prabhu[i] }, { label: "prabhu" })
    neuralNetwork.addData({ image: rihanna[i] }, { label: "rihanna" })
    neuralNetwork.addData({ image: salman[i] }, { label: "salman" })
    neuralNetwork.addData({ image: scarlett[i] }, { label: "scarlett" })
    neuralNetwork.addData({ image: sheryl[i] }, { label: "sheryl" })
    neuralNetwork.addData({ image: whitney[i] }, { label: "whitney" })
  }

  neuralNetwork.normalizeData()
  neuralNetwork.train({
    epochs: 400,
    batchSize: 75
  }, finishedTraining)
}

function finishedTraining() {
  console.log('finished Training!')
  neuralNetwork.save()
}