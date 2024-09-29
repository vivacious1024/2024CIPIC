<template>
  <h3 class="title">SSVEP</h3>

  <div class="button-group">
    <!-- <Button class="mini-block" :label="ppBtn" @click="playpause"></Button> -->
    <!-- <span style="width: 1rem;"></span> -->
    <div>
      <p class="info">
        正在采集:<span style="width: 1rem; color: red"> {{ action }} </span> | 轮次:<span
          style="width: 1rem; color: red"
        >
          {{ block }}
        </span>
        | 刺激次数:<span style="width: 1rem; color: red"> {{ count }} </span> | 采样长度:<span
          style="width: 1rem; color: red"
        >
          {{ ch1.length }}
        </span>
      </p>
    </div>
    <div class="ctrl">
      <input v-model="keyStr" type="text" placeholder="刺激顺序" />
      <Button class="mini-block" label="开始" @click="go"></Button>
      <Button class="mini-block" label="休息" @click="pause"></Button>
      <Button class="mini-block" label="结束" @click="stop"></Button>
    </div>
  </div>
  <p v-if="status != 'eeg'" style="text-align: center">{{ statusText }}</p>
  <div style="background-color: #efefef; padding: 1rem auto; width: 800px; border-radius: 1rem">
    <div id="graphdiv" style="width: 800px; margin: 1rem auto;"></div>
  </div>
</template>

<script>
import 'primevue/resources/themes/lara-light-green/theme.css'
import Button from 'primevue/button'
import { saveAs } from 'file-saver'
import ib2 from '../drivers/ib2/driver.js'
//import filter from '../drivers/ib2/resolve.js'
import { filter } from '../drivers/ib2/resolve.js';

import Dygraph from 'dygraphs'

const SerialPort = window.api.SerialPort

export default {
  name: 'Index',
  components: {
    // eslint-disable-next-line vue/no-reserved-component-names
    Button
  },
  data() {
    return {
      status: 'opening',
      statusText: '',
      device: this.c2o(localStorage.getItem('deviceCode')),
      port: null,
      portPath: localStorage.getItem('portPath'),
      ppBtn: '▶', // ‖‖
      action: false,
      stdStr: 'abcdefghijklmnopqrstuvwxyz1234567890+-*/',
      keyStr: '',
      block: 0,
      count: 0,
      ch1: [],
      ch2: [],
      timer: null,
      flash: 3, //闪烁时长
      data: [],
      dataWindow: [[new Date(), -8388608, 8388607]],
      // dataWindowFiltered: [[new Date(), 0, 0]],
      graph: null
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
    async draw(fp1, fp2) {
      const x = new Date()

      this.dataWindow.push([x, fp1, fp2])
      if (this.dataWindow.length > 160) {
        this.dataWindow.shift()
        // this.dataWindow = [[new Date(), 0, 0]]
      }
      // console.log([x, fp1, fp2]);
      this.graph.updateOptions({ file: this.dataWindow })
    },
    setData(fp1, fp2) {
      if (this.action) {
        this.ch1.push(fp1)
        this.ch2.push(fp2)
        //this.draw(fp1, fp2)
      }
    },
    async setStatus(status, statusText) {
      this.status = status
      this.statusText = statusText
    },
    async connect() {
      ib2.init()
      this.graph = new Dygraph(document.getElementById('graphdiv'), this.dataWindow, {
        labels: ['X', 'Fp1', 'Fp2'],
        // valueRange: [-1.1, 1.1],
        legend: 'always'
      })
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
    async save() {
      const chunkSize = (this.flash + 1) * 500
      const ch1Chunks = []
      const ch2Chunks = []
      this.ch1 = filter(this.ch1)
      this.ch2 = filter(this.ch2)
      for (let i = 0; i < (this.flash + 1) * 500 * 40; i += chunkSize) {
        const chunk1 = this.ch1.slice(i, i + chunkSize)
        const chunk2 = this.ch2.slice(i, i + chunkSize)
        ch1Chunks.push(chunk1)
        ch2Chunks.push(chunk2)
      }
      const ch1List = []
      const ch2List = []
      for (let i = 0; i < 40; i += 1) {
        ch1List.push(ch1Chunks[this.keyStr.indexOf(this.stdStr[i])])
        ch2List.push(ch2Chunks[this.keyStr.indexOf(this.stdStr[i])])
      }
      const blockData = [ch1List, ch2List]
      this.data.push(blockData)
      this.ch1 = []
      this.ch2 = []
      console.log('keystr' + this.keyStr)
    },
    saveMat() {
      // 将四维矩阵转换为 JSON 字符串
      const jsonString = JSON.stringify(this.data)

      // 创建一个 Blob 对象
      const blob = new Blob([jsonString], { type: 'application/json' })

      // 使用 file-saver 保存文件
      saveAs(blob, 'data.json')
    },
    go() {
      if (!this.action) {
        this.block++
        this.timer = setInterval(() => {
          this.count += 1
          console.log(this.ch1.length)
        }, 4000)
        this.action = true
      }
    },
    pause() {
      if (this.action) {
        clearInterval(this.timer)
        this.action = false
        this.save()
      }
    },
    stop() {
      if (!this.action) {
        this.saveMat()
        this.device.stop(this.port)
        this.$router.push('/')
      }
    }
  }
}
</script>

<style scoped>
.button-group {
  justify-content: center;
  margin-bottom: 0.5rem;
}

.ctrl {
}

.info {
  font-size: 1rem;
  margin: 0.5rem;
}

.mini-block {
  height: 8px;
  justify-content: center;
  width: 5rem;
}

.std-block {
  margin: 1rem;
}

.select-inner {
  width: 18rem;
}

.title {
  margin: 0 auto 0.5rem;
}
</style>
