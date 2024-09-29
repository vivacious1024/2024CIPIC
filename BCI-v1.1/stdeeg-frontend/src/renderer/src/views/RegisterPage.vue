<template>
  <div class="box">
    <div class="main">
      <div class="register">
        <span class="title"> 添加被试 </span>
        <el-form ref="registerForm" :model="registerForm" class="form">
          <div class="form-group">
            <span class="iconfont">&#xe624;</span>
            <input
              id="username"
              v-model="registerForm.username"
              type="text"
              prefix-icon="el-icon-user"
              placeholder="用户名"
            />
          </div>
          <br />
          <div class="form-group">
            <span class="iconfont">&#xe908;</span>
            <input
              id="email"
              v-model="registerForm.email"
              type="text"
              prefix-icon="el-icon-user"
              placeholder="邮箱"
            />
          </div>
          <br />
          <button type="primary" class="button" @click="register()">添加</button>
          <div class="navigation">
            <a
              class="register_btn"
              style="background-color: white; color: black; width: 25%; font-size: 15px"
              @click="toLogin()"
              >返回登录</a
            >
            <span> | </span>
            <a
              class="register_btn"
              href="/"
              style="
                background-color: white;
                color: black;
                width: 25%;
                font-size: 15px;
                text-decoration: none;
              "
              >返回首页</a
            >
          </div>
        </el-form>
      </div>
    </div>
  </div>
</template>

<script>
import { Register } from '../api/api'

export default {
  data() {
    return {
      timer: null,
      loading: false,
      registerForm: {
        username: '',
        password: '',
        password2: '',
        email: '',
        code: ''
      }
    }
  },

  methods: {
    register() {
      if (
        typeof this.registerForm.username == 'undefined' ||
        this.registerForm.username == null ||
        this.registerForm.username === ''
      ) {
        //TODO: 弹出提示框
        this.$notify({
          title: '警告',
          message: '用户名不能为空',
          type: 'warning'
        })
        return
      }
      let form_data = new FormData()
      form_data.append('username', this.registerForm.username)
      form_data.append('email', this.registerForm.email)
      Register(form_data).then((res) => {
        if (res.data.result === 0) {
          clearInterval(this.timer)
          this.$router.push('/login')
        } else {
          this.$notify({
            title: '警告',
            message: res.data.message,
            type: 'warning'
          })
        }
      })
    },
    toLogin() {
      this.$router.push('/login')
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

.register {
  width: 440px;
  height: 400px;
  background-color: white;
  align-items: center;
  border-radius: 10px;
  display: flex;
  flex-direction: column;
}

.form {
  height: 50vh;
  width: 100%;
  display: block;
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

.smallword {
  font-size: 20px;
  font-weight: 200;
  font-family: 黑体, serif;
  text-align: right;
}

.button {
  width: 40%;
  height: 50px;
  border-radius: 30px;
  font-size: 20px;
  font-weight: 200;
  font-family: 黑体, serif;
  color: white;
  background-image: linear-gradient(to bottom right, #52927a, #459276);
  margin-top: 20px;
  margin-left: auto;
  margin-right: auto;
}

.button:hover {
  font-weight: bold;
  cursor: pointer;
}

.navigation {
  width: 100%;
  text-align: center;
  margin-top: 20px;
  margin-bottom: 20px;
}

/*.el-form-item ::v-deep .el-form-item__content {
  display: flex;
  justify-content: center;
}*/

a:hover {
  cursor: pointer;
}

input {
  outline-style: none;
  border: 0;
  border-bottom: 1px solid #e9e9e9;
  background-color: transparent;
  height: 20px;
  font-family: sans-serif;
  font-size: 15px;
  color: #445b53;
  font-weight: bold;
}

input:focus {
  border-bottom: 2px solid #445b53;
  background-color: transparent;
  transition: all 0.2s ease-in;
  font-family: sans-serif;
  font-size: 15px;
  color: #445b53;
  font-weight: bold;
}

input:hover {
  border-bottom: 2px solid #445b53;
  background-color: transparent;
  transition: all 0.2s ease-in;
  font-family: sans-serif;
  font-size: 15px;
  color: #445b53;
  font-weight: bold;
}

.register_btn {
  background-color: transparent; /* Green */
  border: none;
  font-size: 12px;
  /* border-radius: 20px;   */
  color: #4e655d;
  font-size: 12px;
  outline: none;
}

.register_btn:hover {
  font-weight: bold;
  cursor: pointer;
}
</style>
