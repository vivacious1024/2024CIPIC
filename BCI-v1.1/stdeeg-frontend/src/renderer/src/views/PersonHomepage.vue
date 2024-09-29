<template>
  <div>
    <div class="navi">
      <NaviBar />
    </div>
    <div class="phpContainer">
      <div class="Info">
        <!--<el-upload
          class="avatar_upload"
          action="#"
          :show-file-list="false"
          :on-success="handleAvatarSuccess"
          :before-upload="beforeAvatarUpload"
          :http-request="uploadPic"
        >
          <img
            class="image-container"
            id="Photo"
            :src="this.imageUrl"
            alt="头像"
            width="170px"
            height="170px"
          />
          <div class="image-black-cover"><i class="el-icon-plus"></i></div>
        </el-upload>-->
        <div class="PersonalInfo">
          <p style="font-size: 20px; color: black; font-weight: bold">
            用户名：{{ username }}
            <span>
              <i class="el-icon-edit" @click="modify" v-show="isSelf">详细资料</i>
              <el-dialog class="info_dialog" v-model="infoDialog" :append-to-body="true">
                <el-descriptions column="1" :label-style="{ 'font-size': '20px' }">
                  <template v-slot:title>
                    <span v-if="infoDialogTitle" style="font-size: 25px">
                      <i class="el-icon-postcard"></i>
                      资料
                    </span>
                    <span v-else style="font-size: 25px">
                      <i class="el-icon-edit-outline"></i>
                      编辑资料
                    </span>
                  </template>
                  <template v-if="infoDialogTitle" v-slot:extra>
                    <el-button type="primary" size="small" @click="modify_info">编辑</el-button>
                  </template>
                  <template v-else v-slot:extra>
                    <el-button type="primary" size="small" @click="modify_confirm()"
                      >确认</el-button
                    >
                    <el-button type="danger" size="small" @click="modify_cancel">取消</el-button>
                    <el-button type="warning" size="small" @click="modify_clear">清空</el-button>
                  </template>
                  <el-descriptions-item label="用户名">
                    <el-input
                      v-model="username"
                      placeholder="请输用户名"
                      :disabled="inputDisabled"
                    ></el-input>
                  </el-descriptions-item>
                  <el-descriptions-item label="邮箱">
                    <el-input
                      v-model="email"
                      placeholder="请输入邮箱"
                      :disabled="inputDisabled"
                    ></el-input>
                  </el-descriptions-item>
                </el-descriptions>
              </el-dialog>
            </span>
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import NaviBar from '@/components/NaviBar.vue'
import index from 'vuex'
import { getInformation } from '@/api/api'
export default {
  name: 'PersonHomepage',
  computed: {
    index() {
      return index
    }
  },
  components: {
    NaviBar
  },
  mounted() {
    this.token = localStorage.getItem('token')
    if (this.token === null) {
      this.$router.push('/login')
    }
    getInformation(this.token).then((res) => {
      if (res.data.result === 0) {
        if (res.data.id === parseInt(this.$route.params.id)) {
          this.isSelf = true
        } else {
          this.isSelf = false
        }
        console.log(res.data)
        console.log(this.isSelf)
        this.email = res.data.email
        this.username = res.data.username
        this.imageUrl = res.data.photo_url_out
      } else {
        this.$notify({
          title: '错误',
          message: '获取用户信息失败',
          type: 'error'
        })
        return
      }
    })
  },
  data() {
    return {
      aid: '',
      token: null,
      is_black: false,
      username: 'username',
      name: 'name',
      institution: 'institution',
      email: '',
      imageUrl: '',
      MidNavIdx: '1',
      Menu1Idx: '1',
      Menu4Idx: '1',
      keywordsInput: '',
      isMasterpieceOnly: false,
      isAuthor: false,
      isManager: false,
      isSelf: false,
      isInterested: false,
      isCancel: false,
      begin1: 0,
      end1: 10,
      begin2: 0,
      end2: 10,
      begin3: 0,
      end3: 10,
      begin4: 0,
      end4: 10,
      author: {
        name: '',
        institution: '',
        email: ''
      },
      scholar_certification: [
        {
          id: '',
          author_id: '',
          username: '',
          send_user_id: '',
          datetime: ''
        }
      ],
      work_certification: [
        {
          id: '',
          work_id: '',
          send_user: '',
          send_user_id: '',
          author_id: '',
          datetime: ''
        }
      ],
      collect_data: [
        {
          paper_id: '',
          paper_name: ''
        }
      ],
      visit_data: [
        {
          id: '',
          work_name: '',
          work_id: '',
          time: ''
        }
      ],
      infoDialog: false,
      infoDialogTitle: true,
      dialogVisible: false,
      inputDisabled: true
    }
  },
  methods: {
    modify() {
      // this.$router.push("/authentication");
      this.infoDialog = true
      console.log(this.infoDialog)
    },
    modify_info() {
      this.infoDialogTitle = false
      this.inputDisabled = false
      console.log(this.infoDialog)
    },
    modify_confirm() {
      this.infoDialogTitle = true
      this.inputDisabled = true
    },
    modify_cancel() {
      this.infoDialogTitle = true
      this.inputDisabled = true
    },
    modify_clear() {
      this.username = ''
      this.name = ''
      this.institution = ''
      this.email = ''
    },
    handleAvatarSuccess(res, file) {
      this.imageUrl = URL.createObjectURL(file.raw)
    },
    beforeAvatarUpload(file) {
      const isJPG = file.type === 'image/jpeg'
      const isLt2M = file.size / 1024 / 1024 < 2

      if (!isJPG) {
        this.$message.error('上传头像图片只能是 JPG 格式!')
      }
      if (!isLt2M) {
        this.$message.error('上传头像图片大小不能超过 2MB!')
      }
      return isJPG && isLt2M
    }
  }
}
</script>

<style scoped>
.navi {
  position: sticky;
  top: 0;
  z-index: 1000;
}

.phpContainer {
  position: relative;
  width: 100%;
  overflow-y: auto;
}

.Info {
  width: 100%;
  background: #e5f0fa;
}

.Info .avatar_upload {
  width: 10%;
  height: 65%;
  float: left;
  position: relative;
  top: 9%;
  left: 10%;
  border-radius: 4px;
  margin-right: 2%;
}

.Info .avatar_upload .image-container {
  position: relative;
}

.Info .avatar_upload .image-black-cover {
  width: 170px;
  height: 170px;
  border-radius: 5px;
  background-color: #121212;
  display: block;
  position: absolute;
  opacity: 0;
  top: 0;
}

.Info .avatar_upload .image-black-cover:hover {
  animation: move 0.6s forwards;
}

@keyframes move {
  0% {
    opacity: 0;
  }
  100% {
    opacity: 0.5;
    transform: scale(1.1);
  }
}

.Info .avatar_upload .image-black-cover .el-icon-plus {
  font-size: 30px;
  position: relative;
  color: white;
  top: 70px;
}

.Info .PersonalInfo {
  height: 65%;
  float: left;
  position: relative;
  top: 15%;
  left: 11%;
  text-align: left;
  line-height: 250%;
}

.Info .PersonalInfo .info_dialog .el_dialog__body {
  background-color: #00b1fd;
}

.Info .PersonalInfo .el-icon-edit {
  margin: 0 0 0 12px;
  font-size: 14px;
  color: #8590a6;
}

.Info .PersonalInfo .el-icon-edit:hover {
  color: black;
}

.Info .PersonalInfo .el-button-interest {
  margin: 0 0 0 12px;
  position: relative;
  top: 3px;
  width: 70px;
  height: 30px;
  font-size: 14px;
  color: #8590a6;
  background-color: transparent;
  border: 1px solid #8590a6;
  border-radius: 4px;
}

.Info .PersonalInfo .el-icon-plus {
  position: relative;
  right: 8px;
  bottom: 4px;
}

.Info .PersonalInfo .el-button-interest:hover {
  color: rgba(0, 0, 0, 0.7);
  border: 1px solid rgba(0, 0, 0, 0.7);
}

.Info .PersonalInfo .el-button-interested {
  margin: 0 0 0 12px;
  position: relative;
  top: 3px;
  width: 70px;
  height: 30px;
  font-size: 14px;
  color: white;
  background: rgba(0, 0, 0, 0.45);
  box-shadow: 0 0 0 2px hsla(0, 0%, 100%, 0.3);
  border-radius: 4px;
}

.Info .PersonalInfo .el-icon-finished {
  position: relative;
  right: 15px;
  bottom: 4px;
}

.Info .PersonalInfo .el-icon-cancel {
  position: relative;
  right: 12.5px;
  bottom: 4px;
}

.Info .PersonalInfo .el-button-interested:hover {
  border: none;
  box-shadow: 0 0 0 2px #8590a6;
}

.MidNav {
  width: 100%;
}

.MidNav .el-menu-demo {
  padding-left: 10%;
}

.MidNav .el-menu-demo .el-menu-item {
  font-size: 16px;
  font-weight: 600;
  font-family:
    pingfang SC,
    helvetica neue,
    arial,
    hiragino sans gb,
    microsoft yahei ui,
    microsoft yahei,
    simsun,
    sans-serif;
  width: 10.3%;
  align-items: center;
  padding: 0 32px 0 32px;
}

.Bottom {
  width: 100%;
  height: 85vh;
  background-color: #f6f6f6;
}

.BottomContent1 {
  margin: 0 7.5%;
  padding: 1% 1% 0;
  width: 82.6%;
  height: 98%;
  background-color: white;
}

.BottomContent1 .el-menu1-demo {
  height: 8%;
  border-bottom: 1px solid #2f3a91;
  margin-bottom: 2%;
}

.BottomContent1 .el-menu1-demo .el-menu-item {
  color: #121212;
  font-size: 14px;
  font-weight: 700;
  font-family:
    pingfang SC,
    helvetica neue,
    arial,
    hiragino sans gb,
    microsoft yahei ui,
    microsoft yahei,
    simsun,
    sans-serif;
  width: 9%;
  height: 100%;
  line-height: 360%;
  border: 1px solid #dcdfe6;
  border-right: none;
  border-bottom: none;
}

.BottomContent1 .el-menu1-demo #item2 {
  border: 1px solid #dcdfe6;
  border-bottom: none;
}

.BottomContent1 .el-menu1-demo .el-menu-item:hover {
  background-color: #2f3a91;
  color: white;
}

.BottomContent1 .el-menu1-demo .el-menu-item.is-active {
  background-color: #2f3a91;
  color: white;
}

.BottomContent1 .keywordSearch {
  width: 25%;
  height: 34px;
  float: left;
}

::v-deep.BottomContent1 .keywordSearch .el-input__inner {
  height: 100%;
  border: 1px solid #dcdfe6;
  border-radius: 20px;
  background-color: #f4f4f4;
}

::v-deep.BottomContent1 .keywordSearch .el-button {
  height: 100%;
  position: relative;
  left: 6%;
  background-color: transparent;
  border: none;
  border-radius: 0 20px 20px 0;
  color: #121212;
  display: inline-block;
}

::v-deep.BottomContent1 .keywordSearch .el-button:hover {
  color: #2f3a91;
}

.BottomContent1 .mp_switch {
  float: left;
  position: relative;
  left: 1%;
  top: 7px;
}

::v-deep.BottomContent1 .mp_switch .el-switch__label {
  color: #646464;
}

::v-deep.BottomContent1 .mp_switch .el-switch__label.is-active {
  color: #2f3a91;
}

.BottomContent1 .dropdown {
  width: 100px;
  float: left;
  position: relative;
  left: 2%;
}

::v-deep.BottomContent1 .dropdown .el-button {
  padding: 10px 0 10px 0;
  width: 90%;
  color: #2f3a91;
  font-size: 12px;
  font-family:
    pingfang SC,
    helvetica neue,
    arial,
    hiragino sans gb,
    microsoft yahei ui,
    microsoft yahei,
    simsun,
    sans-serify;
  text-align: center;
  background-color: #eaebf4;
}

::v-deep.BottomContent1 .el-empty {
  width: 100%;
  height: 500px;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  text-align: center;
  box-sizing: border-box;
}

::v-deep.BottomContent1 .el-empty .el-empty__description p {
  margin: 0;
  font-size: 14px;
  color: #8590a6;
}

.BottomContent2 {
  margin: 0 7.5%;
  padding: 1% 1% 0;
  width: 82.6%;
  height: 98%;
  background-color: white;
}

.BottomContent3 {
  margin: 0 7.5%;
  padding: 1% 1% 0;
  width: 82.6%;
  height: 98%;
  background-color: white;
}

.BottomContent3 .el-table {
  position: relative;
  top: -30px;
}

.BottomContent3 .clear_button {
  width: 80px;
  height: 40px;
  float: right;
  background-color: #67c23a;
  color: white;
  font-size: 16px;
  padding: 12px;
  position: relative;
  top: 10px;
  right: 20px;
  z-index: 500;
}

.BottomContent3 .clear_button:hover {
  filter: brightness(1.2);
}

.BottomContent4 {
  margin: 0 7.5%;
  padding: 1% 1% 0;
  width: 82.6%;
  height: 98%;
  background-color: white;
}

.BottomContent4 .left4 {
  width: 20%;
  height: 100%;
  float: left;
  margin-right: 2%;
}

.BottomContent4 .left4 .el-menu4-demo {
  border: 1px solid #ebebeb;
  border-top: 5px solid #828ad8;
  padding-top: 20px;
  height: 80%;
}

.BottomContent4 .left4 .el-menu4-demo .el-menu-item {
  height: 60px;
  font-family:
    pingfang SC,
    helvetica neue,
    arial,
    hiragino sans gb,
    microsoft yahei ui,
    microsoft yahei,
    simsun,
    sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: #121212;
}

.BottomContent4 .left4 .el-menu4-demo .el-menu-item:hover {
  background-color: #e5f0fa;
}

.BottomContent4 .left4 .el-menu4-demo .el-menu-item.is-active {
  background-color: #e5f0fa;
}

.BottomContent4 .right4_1 {
  width: 78%;
  height: 100%;
  float: left;
}

.BottomContent4 .right4_1 .detail-button {
  width: 80px;
  font-size: 14px;
  background: #e5f0fa;
  opacity: 0.8;
  padding: 12px;
}

.BottomContent4 .right4_1 .detail-button:hover {
  background: #e5f0fa;
  opacity: 1;
}

.check {
  width: 30px;
  height: 30px;
  padding: 5px 5px;
  background-color: #67c23a;
  color: white;
  opacity: 0.8;
  filter: brightness(1);
}

.check:hover {
  background-color: #57ff2a;
  opacity: 2;
  filter: brightness(1.2);
}

.check.is-active {
  border: 1px solid white;
}

.close {
  width: 30px;
  height: 30px;
  padding: 5px 5px;
  background-color: #f56c6c;
  color: white;
  opacity: 0.8;
  filter: brightness(1);
}

.close:hover {
  background-color: #ff4844;
  opacity: 2;
  filter: brightness(1);
}

.close.is-active {
  border: 1px solid white;
}

.BottomContent4 .right4_2 {
  width: 78%;
  height: 100%;
  float: left;
}

.BottomContent4 .right4_2 .detail-button {
  width: 80px;
  font-size: 14px;
  background: #e5f0fa;
  opacity: 0.8;
  padding: 12px;
}

.BottomContent4 .right4_2 .detail-button:hover {
  background: #e5f0fa;
  opacity: 1;
}

.BottomContent4 .right4_3 {
  width: 78%;
  height: 100%;
  float: left;
}

.BottomContent4 .right4_3 .detail-button {
  width: 80px;
  font-size: 14px;
  background: #e5f0fa;
  opacity: 0.8;
  padding: 12px;
}

.BottomContent4 .right4_3 .detail-button:hover {
  background: #e5f0fa;
  opacity: 1;
}
</style>
