import { createApp } from 'vue'
import App from './App.vue'
import { createRouter, createWebHistory } from 'vue-router'
import PrimeVue from 'primevue/config'
import ToastService from 'primevue/toastservice'
import 'primevue/resources/themes/lara-light-green/theme.css'
import './assets/css/icon.css'
import Store from './store'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

const routes = [
  /*{ path: '/', component: () => import('./views/Index.vue') },
  { path: '/about', component: () => import('./views/About.vue') },*/
  {
    path: '/',
    name: '首页',
    redirect: '/main'
  },
  {
    path: '/home',
    name: '主页',
    component: () => import('./views/HomeView.vue'),
    children: [
      {
        path: '/about',
        name: '关于我们',
        component: () => import('./views/About.vue')
      },
      {
        path: '/main',
        name: '首页1',
        component: () => import('./views/Index.vue')
      },

      {
        path: '/settings',
        name: '设置',
        component: () => import('./views/Settings.vue')
      }
    ]
  },
  { path: '/online', component: () => import('./views/Online.vue') },
  /*{ path: '/ssvep', component: () => import('./views/Ssvep.vue') },*/
  { path: '/send', component: () => import('./views/Send.vue') },
  { path: '/big5', component: () => import('./views/Big5.vue') },
  { path: '/aubot', component: () => import('./components/aubot.vue') },
  { path: '/question', component: () => import('./components/question.vue') },
  { path: '/team', component: () => import('./components/team.vue') },

  {
    path: '/php/:id',
    name: '个人主页',
    component: () => import('./views/PersonHomepage.vue')
  },
  {
    path: '/login',
    name: '登录',
    component: () => import('./views/LoginPage.vue')
  },
  {
    path: '/register',
    name: '注册',
    component: () => import('./views/RegisterPage.vue')
  }
]

const app = createApp(App)
const router = createRouter({
  history: createWebHistory(),
  routes
})

app.use(router)
app.use(PrimeVue)
app.use(ToastService)
app.use(Store)
app.use(ElementPlus)
app.mount('#app')
