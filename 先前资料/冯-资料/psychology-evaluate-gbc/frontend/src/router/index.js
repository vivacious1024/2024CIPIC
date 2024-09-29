import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    // {
    //   path: '/',
    //   name: 'home',
    //   component: () => import('../views/HomeView.vue')
    // },
    {
      path: '/',
      name: 'BigFive',
      component: () => import('../views/BigFive.vue')
    },
    {
      path:'/result',
      name: 'BigFiveResult',
      component: () => import('../views/BigFiveResult.vue')
    },
    {
      path: '/real-time',
      name: 'Real Time',
      component: () => import('../views/RealTime.vue')
    }
  ]
})

export default router
