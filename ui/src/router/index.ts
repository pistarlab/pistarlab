import Vue from 'vue'
import VueRouter, { RouteConfig } from 'vue-router'
import Home from '../views/Home.vue'
import Welcome from '../views/Welcome.vue'
import Help from '../views/Help.vue'
import TaskHome from "../views/TaskHome.vue"
import NProgress from "nprogress"
Vue.use(VueRouter)

const routes: Array<RouteConfig> = [
    {
        path: '/',
        name: 'Home',
        component: Home
    },
    {
        path: '/welcome',
        name: 'Welcome',
        component: Welcome
    },
    // {
    //     path: '/dash',
    //     name: 'Dash',
    //     component: Dash
    // }
    // {
    //     path: '/dash',
    //     name: 'dash',
    //     // component: Dash
    //     component: () => import('../views/Dash.vue'),

    // }
    // ,
    // {
    //     path: '/ide',
    //     name: 'ide',
    //     // component: Dash
    //     component: () => import('../views/IDE.vue'),

    // }
    // ,
    // {
    //     path: '/missions/home',
    //     name: 'MissionsHome',
    //     // route level code-splitting
    //     // this generates a separate chunk (about.[hash].js) for this route
    //     // which is lazy-loaded when the route is visited.
    //     component: () => import('../views/MissionsHome.vue')

    // }
    // ,
    // {
    //     path: '/learning/home',
    //     name: 'LearningHome',
    //     // route level code-splitting
    //     // this generates a separate chunk (about.[hash].js) for this route
    //     // which is lazy-loaded when the route is visited.
    //     component: () => import('../views/LearningHome.vue')

    // }
    
    {
        path: '/session/view/:uid',
        component: () => import('../views/SessionView.vue'),
        props: true
    }

    ,
    {
        path: '/episode/view/:uid',
        component: () => import('../views/EpisodeView.vue'),
        props: (route) => ({
            uid: route.params.uid,
            episodeId: route.query.episodeId
        })
    }
    ,
    {
        path: '/session/compare/',
        component: () => import('../views/SessionCompare.vue'),
        props: (route) => ({ uids: route.query.uids })

    }
    ,


    {
        path: '/plugin/home/:category?',
        name: 'PluginHome',
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () => import('../views/PluginHome.vue'),
        props: (route) => ({
            category: route.params.category,
            showWorkspacePlugins: route.query.episodeId,
            managePluginId: route.query.managePluginId
        })

    }

    ,
    {
        path: '/task/specs',
        name: 'TaskSpecs',
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () => import('../views/TaskSpecs.vue'),
        props: true
    }
    ,
    {
        path: '/task/view/:uid',
        name: 'TaskView',
        component: () => import('../views/TaskView.vue'),
        props: true
    }
    ,
    {
        path: '/task/new/agenttask/:uid?',
        name: 'AgentTaskNew',
        component: () => import('../views/AgentTaskNew.vue'),
        props: (route) => ({
            uid: route.params.uid,
            agentUid: route.query.agentUid,

            envSpecId: route.query.envSpecId
        }),

    },
    {
        path: '/task/new/:specId',
        name: 'TaskNew',
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () => import('../views/TaskNew.vue'),
        props: true
    },
    {
        path: '/agent/view/:uid',
        name: 'AgentView',
        component: () => import('../views/AgentView.vue'),
        props: true
    },
    {
        path: '/agent/home',
        name: 'AgentHome',
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () => import('../views/AgentHome.vue')

    },
    {
        path: '/agent/instances',
        name: 'AgentInstances',
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () => import('../views/AgentInstances.vue'),
        props: true
    },


    {
        path: '/agent_spec/:specId',
        name: 'AgentSpecView',
        component: () => import('../views/AgentSpecView.vue'),
        props: true
    }
    ,
    {
        path: '/env/home',
        name: 'EnvHome',
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () => import('../views/EnvHome.vue')
    },
    {
        path: '/component_spec/home',
        name: 'ComponentSpecHome',
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () => import('../views/ComponentSpecHome.vue')
    },

    {
        path: '/component_spec/view/:specId',
        name: 'ComponentSpecView',
        component: () => import('../views/ComponentSpecView.vue'),
        props: true
    },
    {
        path: '/env_spec/view/:specId',
        name: 'EnvSpecView',
        component: () => import('../views/EnvSpecView.vue'),
        props: true
    },
    {
        path: '/data_browser',
        name: 'Data Browser',
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () => import('../views/DataBrowser.vue'),
        props: (route) => ({ path: route.query.path ? route.query.path : "" })
    },

    {
        path: '/help',
        name: 'help',
        component: Help
    }
]

const router = new VueRouter({
    routes
})
router.beforeResolve((to, from, next) => {
    // If this isn't an initial page load.
    if (to.name) {
        // Start the route progress bar.
        NProgress.start()
    }
    next()
})

router.afterEach((to, from) => {
    // Complete the animation of the route progress bar.
    NProgress.done()
})
export default router
