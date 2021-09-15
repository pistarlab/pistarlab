<template>
<div class="page">
    <div class="page-content">

        <h1><i class="fa fa-users"></i> Community Hub </h1>
        <b-alert show variant="info">Coming soon</b-alert>
        <b-nav tabs>
            <b-nav-item :active="$route.path.endsWith('profile')" to="profile">My Profile</b-nav-item>
            <b-nav-item :active="$route.path.endsWith('agents')" to="agents">Find Agents</b-nav-item>
            <b-nav-item :active="$route.path.endsWith('users')" to="users">Find Users</b-nav-item>
        </b-nav>

        <div class="mt-4">
            <router-view></router-view>

        </div>

    </div>
    <HelpInfo contentId="community"></HelpInfo>
</div>
</template>

<script>
import axios from "axios";

import {
    appConfig
} from "../app.config";

import {
    timedeltafordate
} from "../funcs";

export default {
    name: "CommunityHome",
    components: {

    },
    apollo: {
        //

    },
    data() {
        return {
            loading: true,
            data: null,
            message: null

            //

        };
    },
    computed: {
        //

    },
    methods: {
        timedeltafordate,

    },

    created() {

        //
    },
};
</script>

<style >
</style>
