import { createApp } from 'vue'
import App from './App.vue'
import { createRouter, createWebHistory } from 'vue-router'
import PrimeVue from 'primevue/config'
import ToastService from 'primevue/toastservice'
import 'primevue/resources/themes/lara-light-green/theme.css'

const routes = [
  { path: '/', component: () => import('./views/Index.vue') },
  { path: '/about', component: () => import('./views/About.vue') },
  { path: '/online', component: () => import('./views/Online.vue') },
]

const app = createApp(App)
const router = createRouter({
  history: createWebHistory(),
  routes,
})

app.use(router)
app.use(PrimeVue)
app.use(ToastService)
app.mount('#app')
