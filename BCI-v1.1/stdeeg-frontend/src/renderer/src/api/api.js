import service from './service'
// 登陆
export function Login(data) {
  return service({
    method: 'post',
    url: '/user/login',
    data
  })
}
// 注册
export function Register(data) {
  return service({
    method: 'post',
    url: '/user/register',
    data
  })
}
export function GetUsername(data) {
  return service({
    method: 'get',
    url: '/user/get_usernames',
    data
  })
}

export function BigFiveFront(data) {
  return service({
    method: 'get',
    url: '/bigfive/start_front_process',
    data
  })
}

export function BigFiveBack(data) {
  return service({
    method: 'get',
    url: '/bigfive/start_back_process',
    data
  })
}

export function BigFiveAi(data) {
  return service({
    method: 'get',
    url: '/bigfive/start_ai_process',
    data
  })
}

export function BigFiveStop(data) {
  return service({
    method: 'get',
    url: '/bigfive/stop_process',
    data
  })
}

// 发送验证码
export function SendCode(data, token) {
  return service({
    method: 'post',
    url: '/user/send_code',
    data
  })
}
// 确认验证码
export function VerifyCode(data, token) {
  return service({
    method: 'post',
    url: '/user/verify_code',
    data
  })
}

// 修改邮箱
export function ChangeUserEmail(data, token) {
  data = JSON.stringify(data)
  console.log(data, token)
  return service({
    method: 'post',
    url: '/user/change_user_email',
    data,
    headers: {
      Authorization: `Token ${token}`
    }
  })
}

// 修改用户密码
export function ChangeUserPassword(data, token) {
  data = JSON.stringify(data)
  console.log(data)
  return service({
    method: 'post',
    url: '/user/change_user_password',
    data,
    headers: {
      Authorization: `Token ${token}`
    }
  })
}

// 申请成为管理员
export function ApplyAdmin(data, token) {
  return service({
    method: 'post',
    url: '/user/apply_admin',
    data,
    headers: {
      Authorization: `Token ${token}`
    }
  })
}

// 查看浏览历史
export function ShowBrowseHistory(token) {
  return service({
    method: 'get',
    url: '/user/show_browse_history',
    headers: {
      Authorization: `Token ${token}`
    }
  })
}

// 检索相关
export function FuzzySearch(data) {
  data = JSON.stringify(data)
  console.log(data)
  return service({
    method: 'post',
    url: '/academia/fuzzysearch',
    data
  })
}

export function ExactSearch(data) {
  data = JSON.stringify(data)
  console.log(data)
  return service({
    method: 'post',
    url: '/academia/basicsearch',
    data
  })
}

export function MultiSearch(data) {
  data = JSON.stringify(data)
  return service({
    method: 'post',
    url: '/academia/multisearch',
    data
  })
}

export function AuthorFuzzySearch(data) {
  data = JSON.stringify(data)
  return service({
    method: 'post',
    url: '/academia/authorfuzzysearch',
    data
  })
}

// 获取浏览记录
export function GetWorkList(token) {
  return service({
    method: 'get',
    url: '/browhistory/get_work_list',
    headers: {
      Authorization: `Token ${token}`
    }
  })
}

// 添加浏览记录
export function AddBrowHistory(data, token) {
  return service({
    method: 'post',
    url: '/browhistory/add_brow_history',
    data,
    headers: {
      Authorization: `Token ${token}`
    }
  })
}

// 删除浏览记录
export function DeleteBrowHistory(data, token) {
  return service({
    method: 'post',
    url: '/browhistory/delete_brow_history',
    data,
    headers: {
      Authorization: `Token ${token}`
    }
  })
}

// 删除所有浏览记录
export function DeleteAllBrowHistory(token) {
  return service({
    method: 'get',
    url: '/browhistory/delete_all_brow_history',
    headers: {
      Authorization: `Token ${token}`
    }
  })
}

// 获取当前个人所有信息
export function getInformation(token) {
  console.log('ok'+token)
  return service({
    method: 'get',
    url: '/user/get_self_information',
    headers: {
      Authorization: `Token ${token}`
    }
  })
}

// 实名认证
export function Authentication(data, token) {
  return service({
    method: 'post',
    url: '/user/authentication',
    data,
    headers: {
      Authorization: `Token ${token}`
    }
  })
}

//上传头像
export function UploadAvatar(data, token) {
  return service({
    method: 'post',
    url: '/user/upload_avatar',
    data,
    headers: {
      Authorization: `Token ${token}`
    }
  })
}

export function GetAuthor(param) {
  return service({
    method: 'get',
    url: `/academia/get_author`,
    params: {
      author_id: param
    }
  })
}

export function ShowAll() {
  return service({
    method: 'get',
    url: `/message/show_all`
  })
}
