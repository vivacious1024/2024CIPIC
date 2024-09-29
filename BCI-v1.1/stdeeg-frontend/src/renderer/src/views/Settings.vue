<template>
  <div class="path" style="margin-top: 15px">
    <p>存储目录：</p>
    <el-input v-model="textarea" placeholder="请选择文件">
      <template #append>
        <el-button @click="openFile"><span class="iconfont">&#xec17;</span></el-button>
      </template>
    </el-input>
    <input
      id="open"
      type="file"
      name="filename"
      style="display: none"
      webkitdirectory="true"
      @change="changeFile"
    />
  </div>
  <div class="back">
    <el-button type="primary" class="back-button" @click="back">←</el-button>
  </div>
  <div class="foot">
    <el-button type="primary" class="foot-button" @click="save">保存</el-button>
  </div>
</template>

<script>
// 导入组件
export default {
  name: 'Settings',
  components: {},
  data() {
    return {
      textarea: ''
    }
  },
  methods: {
    //选择文件
    openFile() {
      document.getElementById('open').click()
    },
    changeFile() {
      try {
        const fu = document.getElementById('open')
        if (fu == null) return

        console.log('文件夹路径', fu.files[0].webkitRelativePath, fu.files[0].path)
        var webkitRelativePath = fu.files[0].webkitRelativePath
        var path = fu.files[0].path
        var zhenshi = path.substring(
          0,
          path.indexOf(webkitRelativePath.split('/')[webkitRelativePath.split('/').length - 1]) - 1
        )
        console.log('真实路径', zhenshi.replace(/\\/g, '/'))
        this.textarea = zhenshi.replace(/\\/g, '/')
      } catch (error) {
        console.debug('choice file err:', error)
      }
    },
    save() {
      console.log('保存路径', this.textarea)
      const fs = require('fs')
      const path = require('path')

      // 定义保存路径，这里假设保存到用户的文档目录
      const currentWorkingDirectory = process.cwd()
      const parentDirectory = path.dirname(currentWorkingDirectory)
      const savePath = path.join(parentDirectory, 'config.json')

      // 将数据转换为字符串
      const jsonString = JSON.stringify(this.textarea)

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
    back() {
      this.$router.push('/')
    }
  },
  mounted() {
    // 页面加载时，读取文件
    const fs = require('fs')
    const path = require('path')

    // 定义保存路径，这里假设保存到用户的文档目录
    const currentWorkingDirectory = process.cwd()
    const parentDirectory = path.dirname(currentWorkingDirectory)
    const savePath = path.join(parentDirectory, 'config.json')

    fs.readFile(savePath, 'utf8', (err, data) => {
      if (err) {
        console.error('读取文件失败:', err)
      } else {
        console.log('读取文件成功:', data)

        this.textarea = data.replace(/"/g, '')
      }
    })
  }
}
</script>
<style>
.path {
  width: 90%;
  margin: 0 auto;
}
.path p {
  font-family: 'KaiTi', '楷体', serif; /* 尝试使用 'KaiTi' 或 '楷体'，并提供一个后备字体 */
  text-align: left; /* 确保所有.text内的文本都左对齐 */
  margin-bottom: 0;
}
.foot {
  position: absolute;
  bottom: 10%;
  right: 10%;
}
.foot-button {
  background-color: #2db57e;
}
.back {
  position: absolute;
  top: 10%;
  left: 0%;
}
.back-button {
  background-color: #c4e7ed;
  color: #2db57e;
  width: 1rem;
  border: 0;
}
</style>
