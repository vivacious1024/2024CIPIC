<template>
  <h3 class="title">大五人格</h3>

  <div v-if="status == 'on'" class="button-group">
    <!-- <Button class="mini-block" :label="ppBtn" @click="playpause"></Button> -->
    <!-- <span style="width: 1rem;"></span> -->
    <Button class="mini-block" label="结束" @click="stop"></Button>
  </div>
  <p v-if="status != 'on'" style="text-align: center">{{ statusText }}</p>
</template>

<script>
import 'primevue/resources/themes/lara-light-green/theme.css'
import Button from 'primevue/button'
import { BigFiveFront } from '../api/api'
import { BigFiveBack } from '../api/api'
import { BigFiveAi } from '../api/api'
import { BigFiveStop } from '../api/api'

export default {
  name: 'Big5',
  components: {
    // eslint-disable-next-line vue/no-reserved-component-names
    Button
  },
  data() {
    return {
      status: 'opening',
      statusText: '',
      frontUrl: 'E:\\bci\\BCI-v1.0\\big-five\\frontend',
      backUrl: 'E:\\bci\\BCI-v1.0\\big-five\\backend',
      aiUrl: 'E:\\bci\\BCI-v1.0\\big-five\\backend\\python',
      frontProcess: null,
      backProcess: null,
      aiProcess: null
    }
  },
  mounted() {

    this.statusText = `正在尝试打开 大五人格...`
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
    async setStatus(status, statusText) {
      this.status = status
      this.statusText = statusText
    },
    async connect() {
      setTimeout(() => {
        if (this.status === 'opening') {
          this.statusText = `连接失败，三秒后返回`
          setTimeout(() => {
            this.$router.push('/')
          }, 3 * 1000)
        }
      }, 100 * 1000) // 10s
      console.log('connect')
      let response = await this.open()
      if (response) {
        this.status = 'on'
        this.statusText = `打开成功`
      } else {
        this.statusText = `打开失败，三秒后返回`
        setTimeout(() => {
          this.$router.push('/')
        }, 3 * 1000)
        return
      }
      console.log('connected')
    },
    async openFront() {
      let frontResponse = await BigFiveFront()
      if (frontResponse.data.result === 0) {
        console.log('前端服务器已打开')
        return true
      } else {
        this.$notify({
          title: '警告',
          message: '前端服务器打开错误',
          type: 'warning'
        })
        return false
      }
    },
    async openBack() {
      let backResponse = await BigFiveBack()
      if (backResponse.data.result === 0) {
        console.log('后端服务器已打开')
        return true
      } else {
        this.$notify({
          title: '警告',
          message: '后端服务器打开错误',
          type: 'warning'
        })
        return false
      }
    },
    async openAi() {
      let aiResponse = await BigFiveAi()
      if (aiResponse.data.result === 0) {
        console.log('算法服务器已打开')
        return true
      } else {
        this.$notify({
          title: '警告',
          message: '算法服务器打开错误',
          type: 'warning'
        })
        return false
      }
    },
    async open() {
      let front = await this.openFront()
      let back = await this.openBack()
      let ai = await this.openAi()
      if (front && back && ai) {
        return true
      } else {
        return false
      }
    },
    async stop() {
      this.statusText = `正在关闭 大五人格...`
      BigFiveStop()
      this.$router.push('/')
    }
  }
}
</script>
