<template>

  
  <Toast></Toast>
  <h3 class="title">连接你的脑机接口</h3>

  <div class="p-float-label std-block">
    <Dropdown v-model="device" :options="deviceList" filter showClear optionLabel="name" class="w-full md:w-14rem select-inner"></Dropdown>
    <label>选择脑机接口型号</label>
  </div>

  <div class="p-float-label std-block">
    <Dropdown v-model="port" :options="portList" filter showClear optionLabel="name" class="w-full md:w-14rem select-inner"></Dropdown>
    <label>选择端口</label>
  </div>

  <Button class="std-block" label="进入采集模式" @click="online"></Button>

  <div class="p-float-label std-block">
    <Dropdown v-model="task" :options="taskList" filter showClear optionLabel="name" class="w-full md:w-14rem select-inner"></Dropdown>
    <label>选择任务</label>
  </div>

  <Button class="std-block" label="进入任务模式" @click="execute"></Button>

</template>

<script>
import Dropdown from 'primevue/dropdown';
import 'primevue/resources/themes/lara-light-green/theme.css'
import Button from 'primevue/button';
import Toast from 'primevue/toast';

const SerialPort = window.api.SerialPort;

export default {
  name: 'Index',
  components: {
    Dropdown, Button, Toast
  },
  async mounted() {
    this.scanInterval = setInterval(() => {
      this.getPortList();
    }, 1000);
  },
  data() {
    return {
      scanInterval: null,
      device: null,
      deviceList: [
        { name: 'I-BRAIN 便携式两导', code: 'ib2' },
      ],
      port: null,
      portList: [],
      task: null,
      taskList: [
        { name: '神经指纹身份认证', code: 'nfa' }, // Neural Fingerprint Authentication
      ],
    };
  },
  methods: {
    async getPortList() {
      await SerialPort.list().then((ports, err) => {
        if(err) {
          console.error(err);
          return;
        }
        console.log(ports);
        this.portList = ports.map((port) => {
          return {
            name: port.friendlyName, // `${port.path} ${port.manufacturer.startsWith('(')?'':'('}${port.manufacturer}${port.manufacturer.endsWith(')')?'':')'}`,
            path: port.path
          }
        });
      });
    },
    online() {
      if(this.device && this.port?.path) {
        localStorage.setItem('deviceCode', this.device?.code);
        localStorage.setItem('portPath', this.port?.path);
        clearInterval(this.scanInterval);
        this.$router.push('/online');
      } else {
        this.$toast.add({
          severity: 'error',
          summary: '错误',
          detail: '请选择脑机接口型号和端口',
          life: 3000
        });
      }
    },
    execute() {
      if(this.device && this.port?.path) {
        if(this.task) {
          localStorage.setItem('task', JSON.stringify(this.task));
          this.$router.push('/execute');
        } else {
          this.$toast.add({
            severity: 'error',
            summary: '错误',
            detail: '请选择任务',
            life: 3000
          });
        }
      } else {
        this.$toast.add({
          severity: 'error',
          summary: '错误',
          detail: '请选择脑机接口型号和端口',
          life: 3000
        });
      }
    },
  }
};
</script>

<style scoped>
.title {
  margin: 0 auto 2rem;
}
.std-block {
  margin: 1rem;
}
.select-inner {
  width: 18rem;
}
</style>