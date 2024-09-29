import { validate, getData } from './resolve.js'

//设备名称、代码、波特率、采样率、通道数和通道名称
const param = {
  name: 'I-BRAIN 便携式两导',
  code: 'ib2',
  buadRate: 115200,
  sample: 500,
  channel: 2,
  channelName: ['Fp1', 'Fp2']
}

// bufer
const buffer = Buffer.allocUnsafe(2048 * 1024) // 2MB
let bufSize = 0
let status = 'bluetooth'
let setStatus = null
let setData = null
// blue string
let blueMsg = ''

const init = () => {
  buffer.fill(0)
  bufSize = 0
  status = 'bluetooth'
  setStatus = null
  setData = null
  blueMsg = ''
}

const blueSolve = async (port, data) => {
  const msg = data.toString()
  blueMsg += msg
  // console.log(msg);
  if (msg == '+IM_MASTER\r\n') {
    blueMsg = ''
    port.write('AT+SCAN=1,5\r\n')
    setStatus('bluetooth', '开始扫描设备')
  } else if (msg == '+IM_READY\r\n') {
    const regex = /YN_.+/g
    const found = Array.from(new Set(blueMsg.match(regex)))
    console.log('found:', found)
    setStatus('bluetooth', '发现设备，开始连接')
    blueMsg = ''
    port.write(`AT+CONN=0,${found[0].split('YN_')[1]}\r\n`)
  } else if (msg == '+IM_CONNECT\r\n') {
    // 然后就自动开始吐数据了
    blueMsg = ''
    status = 'eeg'
    bufSize = 0
    setStatus('eeg', '连接成功！开始接收数据')
  }
}

// eslint-disable-next-line no-unused-vars
const fileAppend = async () => {
  // output eeg file with append mode
}

const connect = async (SerialPort, portPath) => {
  const port = new SerialPort(
    {
      path: portPath,
      baudRate: param.buadRate
    },
    (err) => {
      if (err) {
        console.error('Error:', err.message)
        return
      }
    }
  )

  port.on('open', () => {
    console.log('port open')
  })

  port.on('data', (data) => {
    // console.log('Data:', data);
    // put into buffer
    for (let i of data) {
      if (i == 10 && status == 'bluetooth') {
        // '\n'
        buffer[bufSize++] = 10
        blueSolve(port, buffer.subarray(0, bufSize))
        bufSize = 0
        continue
      } else if (status == 'eeg') {
        if (bufSize >= 208) {
          // 208 must have an intact 104
          // just seek for 104, and get data, then slice it
          for (let i = 0; i < bufSize; ++i) {
            if (validate(buffer, i)) {
              const data = getData(buffer.subarray(i, i + 104))
              // console.log(data);
              // fetch python

              for (let j = 0; j < 16; ++j) {
                setData(data.ch1[j], data.ch2[j])
              }
              //setData(data.ch1[0], data.ch2[0]);
              buffer.set(buffer.subarray(i + 104, bufSize))
              bufSize = bufSize - 104 - i
              break
            } else {
              console.log('error data')
            }
          }
        }
      }
      buffer[bufSize++] = i
    }
  })
  return port
}

const start = (port, func1, func2) => {
  port.write('AT+ROLE=0\r\n', (err) => {
    if (err) {
      return console.log('Error on write: ', err.message)
    }
    console.log('start message written')
  })
  setStatus = func1
  setData = func2
}

// eslint-disable-next-line no-unused-vars
const play = (port) => {
  // bug
  port.write(new Uint8Array([0x55, 0xaa, 0x01, 0x55, 0xaa])) // play
  console.log('play invoked')
}

// eslint-disable-next-line no-unused-vars
const pause = (port) => {
  // bug
  port.write(new Uint8Array([0x55, 0xaa, 0x00, 0x55, 0xaa])) // pause
  console.log('pause invoked')
}

const stop = async (port) => {
  await port.write(
    new Uint8Array([
      0xaa, 0xaa, 0x02, 0x0a, 0x64, 0x69, 0x73, 0x63, 0x6f, 0x6e, 0x6e, 0x65, 0x63, 0x74, 0xd5
    ])
  )
  await port.close()
  console.log('stop invoked')
}

const disconnect = () => {
  // reserve
}

export default { init, param, connect, start, stop, disconnect }
