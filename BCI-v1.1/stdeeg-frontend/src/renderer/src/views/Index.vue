<template>
  <Toast></Toast>
  <h1>欢迎使用便携式两导设备脑机接口平台</h1>
  <h3 class="title">连接你的脑机接口</h3>

  <div class="bg"></div>

  <div class="flex-container">
    <!-- 你的.std-block元素将在这里 -->
    <div class="p-float-label std-block">
      <Dropdown
        v-model="device"
        :options="deviceList"
        filter
        show-clear
        option-label="name"
        class="w-full md:w-14rem select-inner"
      >
      </Dropdown>
      <label>选择脑机接口型号</label>
    </div>

    <div class="p-float-label std-block">
      <Dropdown
        v-model="port"
        :options="portList"
        filter
        show-clear
        option-label="name"
        class="w-full md:w-14rem select-inner"
      ></Dropdown>
      <label>选择端口</label>
    </div>

    <Button class="std-block" label="采集模式" @click="online"></Button>

    <div class="p-float-label std-block task">
      <Dropdown
        v-model="task"
        :options="taskList"
        filter
        show-clear
        option-label="name"
        class="w-full md:w-14rem task-select-inner"
      ></Dropdown>
      <label>选择任务</label>
      <Button class="task-button" label="进入" @click="execute"></Button>
    </div>
  </div>
</template>

<script>
import Dropdown from 'primevue/dropdown'
import 'primevue/resources/themes/lara-light-green/theme.css'
import Button from 'primevue/button'
import Toast from 'primevue/toast'

const SerialPort = window.api.SerialPort

export default {
  name: 'Index',
  components: {
    Dropdown,
    // eslint-disable-next-line vue/no-reserved-component-names
    Button,
    Toast
  },
  data() {
    return {
      scanInterval: null,
      device: null,
      deviceList: [{ name: 'I-BRAIN 便携式两导', code: 'ib2' }],
      port: null,
      portList: [],
      task: null,
      taskList: [
        { name: '大五人格', code: 'big5' }, // Neural Fingerprint Authentication
        /*{ name: '预赛', code: 'ssvep' },*/
        { name: '决赛', code: 'send' }
      ]
    }
  },
  async mounted() {
    this.scanInterval = setInterval(() => {
      this.getPortList()
    }, 1000)
  },
  methods: {
    async getPortList() {
      await SerialPort.list().then((ports, err) => {
        if (err) {
          console.error(err)
          return
        }
        console.log(ports)
        this.portList = ports.map((port) => {
          return {
            name: port.friendlyName, // `${port.path} ${port.manufacturer.startsWith('(')?'':'('}${port.manufacturer}${port.manufacturer.endsWith(')')?'':')'}`,
            path: port.path
          }
        })
      })
    },
    online() {
      if (this.device && this.port?.path) {
        localStorage.setItem('deviceCode', this.device?.code)
        localStorage.setItem('portPath', this.port?.path)
        clearInterval(this.scanInterval)
        this.$router.push('/online')
      } else {
        this.$toast.add({
          severity: 'error',
          summary: '错误',
          detail: '请选择脑机接口型号和端口',
          life: 3000
        })
      }
    },
    execute() {
      if (this.device && this.port?.path) {
        if (this.task) {
          localStorage.setItem('deviceCode', this.device?.code)
          localStorage.setItem('portPath', this.port?.path)
          clearInterval(this.scanInterval)
          localStorage.setItem('task', JSON.stringify(this.task))
          this.$router.push(this.task?.code)
        } else {
          this.$toast.add({
            severity: 'error',
            summary: '错误',
            detail: '请选择任务',
            life: 3000
          })
        }
      } else {
        this.$toast.add({
          severity: 'error',
          summary: '错误',
          detail: '请选择脑机接口型号和端口',
          life: 3000
        })
      }
    }
  }
}
</script>

<style scoped>
.flex-container {
  display: flex;
  flex-wrap: wrap;
  /* 允许flex项目换行 */
  justify-content: flex-start;
  /* 项目在主轴上左对齐 */
  /* 抵消第一个项目的左边距 */
  width: calc(100%);
  /* 抵消左边距的负值，使容器宽度正确 */
}

.title {
  margin: 0 auto 2rem;
}

.std-block {
  margin: 2rem 3rem;
  /* 为每个块设置宽度为容器宽度的一半减去间距 */
  flex: 0 0 calc(50% - 6rem);
  /* 确保块之间有适当的间距 */
  max-width: calc(50% - 6rem);
  /* 使块在flex容器中垂直居中 */
  display: flex;
  align-items: center;
}

.select-inner {
  width: 100%;
}

.task {
  /*display: inline-block;*/
}

.task-button {
  width: 30%;
}

.task-select-inner {
  width: 70%;
}
</style>
