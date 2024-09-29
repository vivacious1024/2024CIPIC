<template>
  <Toast></Toast>
  <div class="box">
    <div class="main">
      <div class="login">
        <span class="title"> 选择被试 </span>
        <div class="form">
          <div class="p-float-label list">
            <Dropdown
              v-model="user"
              :options="users"
              filter
              show-clear
              option-label="name"
              class="w-full md:w-14rem select-inner"
            >
            </Dropdown>
            <label>选择被试</label>
            <Button class="select-button" @click="login()">选择用户</Button>
          </div>
          <br />
          <div class="navigation">
            <a
              class="login_btn"
              href="/register"
              style="background-color: white; color: black; width: 25%; font-size: 15px"
              >添加被试</a
            >
            <span> | </span>
            <a
              class="login_btn"
              href="/"
              style="background-color: white; color: black; width: 25%; font-size: 15px"
              >返回首页</a
            >
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import Dropdown from 'primevue/dropdown'
import 'primevue/resources/themes/lara-light-green/theme.css'
import Button from 'primevue/button'
import Toast from 'primevue/toast'
import { Login } from '../api/api'
import { GetUsername } from '../api/api'
import { getInformation } from '@/api/api'
export default {
  components: {
    // 在这里注册想要在本页面中使用的子组件
    Dropdown,
    Button,
    Toast
  },
  data() {
    return {
      user: null,
      users: [],
      userSelected: '',
      textarea: '',
      token: ''
    }
  },
  mounted() {
    this.getusername()
  },
  methods: {
    async login() {
      if (typeof this.user == 'undefined' || this.user == null || this.user === '') {
        //TODO: 弹出提示框
        this.$notify({
          title: '警告',
          message: '用户名不能为空',
          type: 'warning'
        })
        return
      }
      let form_data = new FormData()
      form_data.append('username', this.user.name)
      //TODO: 发送登录请求
      const loginResponse = await Login(form_data)
      if (loginResponse.data.result === 0) {
        localStorage.setItem('token', loginResponse.data.token)
      } else {
        this.$notify({
          title: '警告',
          message: '用户名错误',
          type: 'warning'
        })
      }
      this.token = localStorage.getItem('token')

      const informationResponse = await getInformation(this.token)
      if (informationResponse.data.result === 0) {
        console.log(informationResponse.data)
        this.userSelected = informationResponse.data.username
      } else {
        this.$notify({
          title: '错误',
          message: '获取用户信息失败',
          type: 'error'
        })
        return
      }
      const fs = require('fs')
      const path = require('path')

      // 定义保存路径，这里假设保存到用户的文档目录
      const currentWorkingDirectory = process.cwd()
      const parentDirectory = path.dirname(currentWorkingDirectory)
      const savePath = path.join(parentDirectory, 'user.json')

      // 将数据转换为字符串
      const jsonString = JSON.stringify(this.userSelected)

      // 保存文件到固定位置
      fs.writeFile(savePath, jsonString, (err) => {
        if (err) {
          console.error('保存文件失败:', err)
        } else {
          console.log('文件已保存到:', savePath)
        }
      })
      this.$router.push('/')

    },
    getusername() {
      GetUsername().then((res) => {
        this.users = res.data.usernames.map((username) => ({ name: username, selected: false }))
      })
    }
  }
}
</script>

<style scoped>
.box {
  position: absolute;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  justify-content: center;
  align-items: center;
  overflow: hidden;
  margin: 0;
}

.main {
  width: 100vw;
  height: 100vh;
  background-image: url('../assets/loginBG.png');
  display: flex;
  justify-content: center;
  align-items: center;
}

.list {
  margin: 2rem 3rem;
  /* 为每个块设置宽度为容器宽度的一半减去间距 */
  flex: 0 0 calc(100% - 6rem);
  /* 确保块之间有适当的间距 */
  max-width: calc(100% - 6rem);
  /* 使块在flex容器中垂直居中 */
  display: flex;
  align-items: center;
}

.login {
  width: 480px;
  height: 450px;
  background-color: white;
  align-items: center;
  border-radius: 10px;
  display: flex;
  flex-direction: column;
}

.login_btn {
  background-color: transparent; /* Green */
  border: none;
  font-size: 12px;
  /* border-radius: 20px;   */
  color: #4e655d;
  font-size: 12px;
  outline: none;
}

.login_btn:hover {
  font-weight: bold;
  cursor: pointer;
}

.form {
  height: 50vh;
  width: 100%;
  margin-top: 20px;
}

.title {
  width: 100%;
  margin-top: 20px;
  margin-left: 40px;
  margin-bottom: 40px;
  text-align: left;
  color: black;
  font-size: 20px;
  font-weight: 800;
  text-decoration: underline #0066ff;
  text-underline-offset: 20px;
  text-decoration-thickness: 3px;
}

.select-inner {
  width: 70%;
}

.select-button {
  width: 30%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.smallword {
  font-size: 20px;
  font-weight: 200;
  font-family: 黑体, serif;
  margin-right: 20px;
}

.button {
  width: 80%;
  height: 50px;
  border-radius: 30px;
  font-size: 20px;
  font-weight: 200;
  font-family: 黑体, serif;
  color: white;
  background-image: linear-gradient(to bottom right, #2799d3, #004daa);
  margin-top: 20px;
  margin-left: auto;
  margin-right: auto;
}

.navigation {
  width: 100%;
  text-align: center;
  margin-top: 20px;
  margin-bottom: 20px;
}

a {
  text-decoration: none;
}
</style>
