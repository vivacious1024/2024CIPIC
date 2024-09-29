<template lang="html">
  <div class="navi">
    <div class="navi-inner">
      <img class="logo" alt="logo" src="../assets/icon.png" @click="gotoMain" />
      <div
        v-for="(naviUnit, index) in naviUnits"
        :key="index"
        class="navi-item"
        :class="{ active: activeIndex === index }"
        @click="naviUnit.submenu ? null : changeActive(index)"
        @mouseover="showSubmenu(index)"
        @mouseleave="hideSubmenu(index)"
      >
        <div class="navi-item-link">
          <router-link :to="naviUnit.link_to">{{ naviUnit.content }}</router-link>
        </div>
        <div v-if="naviUnit.submenu && activeSubmenu === index" class="submenu">
          <div
            v-for="(subItem, subIndex) in naviUnit.submenu"
            :key="subIndex"
            class="submenu-item"
            @click="changeActive(index)"
          >
            <router-link :to="subItem.link_to">{{ subItem.content }}</router-link>
          </div>
        </div>
      </div>
      <div v-show="isLogin" class="photo">
        <el-dropdown placement="bottom" @command="handleCommand">
          <el-avatar :size="35" :src="photoURL"></el-avatar>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="a"><i class="el-icon-house"></i>主页</el-dropdown-item>
              <el-dropdown-item divided command="d"
                ><i class="el-icon-setting"></i>设置</el-dropdown-item
              >
              <el-dropdown-item command="e"
                ><i class="el-icon-switch-button"></i>退出</el-dropdown-item
              >
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
      <div v-show="!isLogin" class="loginButton">
        <el-button class="el-loginButton" @click="gotoLogin">选择被试</el-button>
      </div>
    </div>
  </div>
</template>

<script>
import { getInformation } from '../api/api'
export default {
  data() {
    return {
      token: null,
      activeIndex: null,
      activeSubmenu: null,
      naviUnits: [
        { content: '首页', link_to: '/main' },
        { content: '关于我们', link_to: '/about' },
        { content: '设置', link_to: '/settings' }
      ],
      input1: '',
      select: '1',
      isLogin: false,
      id: '',
      photoURL: ''
    }
  },
  mounted() {
    this.token = localStorage.getItem('token')
    if (this.token != null) {
      this.isLogin = true
    } else {
      return
    }
    console.log('token' + this.token)
    console.log('id' + this.id)
    getInformation(this.token).then((res) => {
      if (res.data.result === 0) {
        this.id = res.data.id
        this.photoURL = res.data.photo_url_out
      } else {
        this.$notify({
          title: '错误',
          message: '获取用户信息失败',
          type: 'error'
        })
        return
      }
      console.log('id' + this.id)
    })
  },
  created() {
    const currentRouteIndex = this.naviUnits.findIndex((unit) => unit.link_to == this.$route.path)
    this.activeIndex = currentRouteIndex !== -1 ? currentRouteIndex : null

    this.$watch('$route', () => {
      const currentRouteIndex = this.naviUnits.findIndex(
        (unit) => unit.link_to === this.$route.path
      )
      this.activeIndex = currentRouteIndex !== -1 ? currentRouteIndex : null
    })
  },
  methods: {
    changeActive(index) {
      this.activeIndex = index
      console.log(this.activeIndex)
    },
    showSubmenu(index) {
      this.activeSubmenu = index
    },
    hideSubmenu(index) {
      this.activeSubmenu = -1
    },
    gotoMain() {
      if (this.$route.path !== '/main') {
        this.$router.push('/main')
      }
    },
    gotoLogin() {
      this.$router.push('/login')
    },
    handleCommand(command) {
      if (command === 'a' && this.$route.path !== '/php/' + this.id) {
        this.$router.push('/php/' + this.id)
      }
      if (command === 'b') {
        // handle command b
      }
      if (command === 'c') {
        // handle command c
      }
      if (command === 'd' && this.$route.path !== '/settings') {
        this.$router.push('/settings')
      }
      if (command === 'e') {
        this.isLogin = false
        localStorage.removeItem('token')
        this.token = null

        const fs = require('fs')
        const path = require('path')

        // 定义保存路径，这里假设保存到用户的文档目录
        const currentWorkingDirectory = process.cwd()
        const parentDirectory = path.dirname(currentWorkingDirectory)
        const savePath = path.join(parentDirectory, 'user.json')

        // 将数据转换为字符串
        const jsonString = JSON.stringify('temp')

        // 保存文件到固定位置
        fs.writeFile(savePath, jsonString, (err) => {
          if (err) {
            console.error('保存文件失败:', err)
          } else {
            console.log('文件已保存到:', savePath)
          }
        })

        if (this.$route.path !== '/main') {
          this.$router.push('/main')
        }
      }
    }
  }
}
</script>

<style lang="css">
.navi {
  width: 100%;
  background-color: #1e4d3c;
  height: 60px;
  margin-left: 0%;
  padding: 0;
}

.navi-inner {
  width: 86%;
  height: 100%;
  position: relative;
  left: 7%;
}

.navi-inner .logo {
  display: inline-block;
  width: 60px;
  height: 60px;
  float: left;
  opacity: 0.75;
}

.navi-inner .navi-item {
  display: inline-block;
  float: left;
  position: relative;
  height: 60px;
  left: 10px;
}

.navi-inner .navi-item a {
  display: inline-block;
  color: #d5d8e9;
  line-height: 60px;
  text-decoration: none;
  padding: 0 15px;
}

.navi-inner .navi-item:hover {
  background-color: #52927a;
}

.navi-inner .navi-item:hover a {
  color: #fff;
}

.navi-inner .navi-item::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 5px;
  background-color: #ddbb34;
  opacity: 0;
}

.navi-inner .navi-item.active::after {
  opacity: 1;
}

.navi-inner .navi-item.active {
  background-color: #52927a;
}

.navi-inner .navi-item.active a {
  font-weight: bold;
  color: #fff;
}

.navi-inner .navi-item .submenu {
  left: 10px;
  background-color: #fff;
  opacity: 0.7;
}

.navi-inner .navi-item .submenu .submenu-item {
  height: 40px;
}

.navi-inner .navi-item .submenu a {
  align-items: center;
  justify-content: center;
  color: #1524ae;
  line-height: 40px;
  text-decoration: none;
  padding: 0 5px;
}

.navi-search {
  width: calc(550px * 0.5); /* 原宽度的60% */
  float: left;
  position: relative;
  top: 10px;
  left: calc(30px * 0.5);
  border-radius: 20px 0 0 20px;
  margin-left: calc(240px * 0.5);
}

.navi-search .el-select {
  float: left;
  width: calc(128px * 0.5);
  height: 40px;
}

.navi-search .el-select .el-input__inner {
  background-color: #ffffff33;
  color: #ffffffcc;
  border-radius: 20px 0 0 20px;
  border: 1px solid transparent;
  padding: 10px 10px 10px 15px;
  font-size: 12px;
}

.navi-search .el-select:hover .el-input__inner {
  border: 1px solid transparent;
}

.navi-search .el-select .el-input__inner:focus {
  color: #2f3a91;
  background-color: #ffffff;
  border: 1px solid transparent;
}

.navi-search .el-select .el-input.is-focus .el-input__inner {
  border: 1px solid transparent;
}

.navi-search .navi-input {
  float: left;
  width: calc(416px * 0.5);
}

.navi-search .navi-input > .el-input__inner {
  float: left;
  width: calc(416px * 0.5);
  color: #ffffffcc;
  background-color: #ffffff33;
  font-size: 12px;
  border-radius: 0 20px 20px 0;
  border: 1px solid transparent;
}

.navi-search .navi-input > .el-input__inner:focus {
  color: #2f3a91;
  background-color: #ffffff;
}

.navi-search .navi-input .el-button {
  height: 40px;
  position: relative;
  left: calc(5px * 0.5);
  background-color: transparent;
  border: 1px solid transparent;
  border-radius: 0 20px 20px 0;
  color: #bbbbbb;
}

/*.navi-search .navi-input .el-button:hover {*/
/*  color: #2f3a91;*/
/*}*/

.navi-inner .photo {
  display: inline-block;
  float: right;
  position: relative;
  height: 60px;
  width: 60px;
}

.navi-inner .photo .el-avatar {
  position: relative;
  top: 12.5px;
}

.navi-inner .loginButton {
  display: inline-block;
  float: right;
  position: relative;
  height: 30px;
  width: 70px;
  top: 10px;
  /*color: #ffffff;
  border: 1px solid #ffffff;
  border-radius: 4px;*/
}

.navi-inner .loginButton .el-loginButton {
  background: transparent;
  color: #ffffffcc;
  border: 1px solid #ffffffcc;
  border-radius: 4px;
  position: center;
  cursor: pointer;
}

.navi-inner .loginButton .el-loginButton:hover {
  color: white;
  border: 1px solid white;
  cursor: pointer;
}
</style>
<!---可能的问题：activesubmenu没初始化-->
