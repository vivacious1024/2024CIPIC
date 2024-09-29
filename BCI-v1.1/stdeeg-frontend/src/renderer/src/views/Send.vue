<template>
  <h3 class="title">决赛</h3>

  <div v-if="status == 'eeg'" class="button-group">
    <!-- <Button class="mini-block" :label="ppBtn" @click="playpause"></Button> -->
    <!-- <span style="width: 1rem;"></span> -->
    <Button class="mini-block" label="■" @click="stop"></Button>
  </div>
  <p v-if="status != 'eeg'" style="text-align: center">{{ statusText }}</p>

  <div class="back">
    <el-button type="primary" class="back-button" @click="back">←</el-button>
  </div>
</template>

<script>
import 'primevue/resources/themes/lara-light-green/theme.css'
import Button from 'primevue/button'
import ib2 from '../drivers/ib2/driver.js'
// import { filter } from '../drivers/ib2/resolve.js';

const SerialPort = window.api.SerialPort
const { ipcRenderer } = require('electron')
ipcRenderer.on('tcp-data-received', (event, message) => {
  console.log('Data received from TCP server:', message)
  // 更新 UI 或处理数据
})
export default {
  name: 'Index',
  components: {
    // eslint-disable-next-line vue/no-reserved-component-names
    Button
  },
  data() {
    return {
      count: 0,
      buf: Buffer.alloc(12*40),
      status: 'opening',
      statusText: '',
      device: this.c2o(localStorage.getItem('deviceCode')),
      port: null,
      portPath: localStorage.getItem('portPath'),
      ws: null, // 客户端
      ppBtn: '▶' // ‖‖
    }
  },
  mounted() {
    this.statusText = `正在尝试打开 ${this.portPath}...`
    this.connect()

    // It sucks that these things aren't objects, and we need to store state in window.
    // setInterval(function() {
    //   var x = new Date();  // current time
    //   var y1 = Math.random()-.5;
    //   var y2 = Math.random()-.5;
    //   data.push([x, y1, y2]);
    //   g.updateOptions( { 'file': data } );
    // }, 1000);
  },
  methods: {
    c2o(code) {
      switch (code) {
        case 'ib2':
          return ib2
        default:
          return null
      }
    },
    async setData(fp1, fp2) {
      const numFp1 = Number(fp1)
      const numFp2 = Number(fp2)

      const buffer = Buffer.alloc(12)
      buffer.writeFloatLE(numFp1, 0)
      buffer.writeFloatLE(numFp2, 4)
      buffer.writeFloatLE(0, 8)

      if (this.count == 0) {
        console.log(new Date());
      }
      if (this.count < 40) {
        buffer.copy(this.buf, this.count * 12)
        this.count++
      } else {
        this.count = 0
        ipcRenderer.send('send-tcp-data', this.buf)
        console.log(this.buf)
      }
    },
    async setStatus(status, statusText) {
      this.status = status
      this.statusText = statusText
    },
    async connect() {
      ib2.init()
      setTimeout(() => {
        if (this.status === 'opening') {
          this.statusText = `连接失败，三秒后返回`
          setTimeout(() => {
            this.$router.push('/')
          }, 3 * 1000)
        }
      }, 10 * 1000) // 10s

      this.port = await this.device.connect(SerialPort, this.portPath)

      if (this.port) {
        this.status = 'opened'
        this.statusText = `串口打开成功，正在握手...`
      } else {
        this.statusText = `串口打开失败，三秒后返回`
        setTimeout(() => {
          this.$router.push('/')
        }, 3 * 1000)
        return
      }

      await this.device.start(this.port, this.setStatus, this.setData)
    },
    // playpause() {
    //   if(this.ppBtn == '▶') {
    //     this.ppBtn = '‖‖';
    //     this.device.play(this.port);
    //   } else {
    //     this.ppBtn = '▶';
    //     this.device.pause(this.port);
    //   }
    // },
    stop() {
      this.device.stop(this.port)
      this.client?.end()
      this.$router.push('/')
    },
    back() {
      if (this.port) {
        this.device.stop(this.port)
      }
      this.client?.end()
      this.$router.push('/')
    }
  }
}
</script>

<style scoped>
.title {
  margin: 0 auto 0.5rem;
}

.std-block {
  margin: 1rem;
}

.mini-block {
  float: left;
  justify-content: center;
  width: 1rem;
}

.button-group {
  display: inline-flex;
  justify-content: center;
  margin-bottom: 0.5rem;
}

.select-inner {
  width: 18rem;
}

.back {
  position: absolute;
  top: 2%;
  left: 2%;
}

.back-button {
  background-color: #c4e7ed;
  border: 0;
}
</style>
