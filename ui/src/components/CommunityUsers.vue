<template>
<div>
    <div v-if="items && items.length>0">
        <b-row>
            <b-col>User Id</b-col>

            <b-col> Join Date</b-col>
            <b-col></b-col>
        </b-row>
        <hr />
        <div v-for="(user, i) in items" v-bind:key="i">
            <b-row>
                <b-col> {{user.user_id}}</b-col>
                <b-col> {{user.created}}</b-col>
                <b-col></b-col>
            </b-row>
        </div>
    </div>
    <div v-else-if="loading && !items">
        Loading...
    </div>
    <div v-else>
        No Users Found
        <br />
        {{message}}
    </div>
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
    name: "UsersOnline",
    components: {
        //
    },
    apollo: {
        //

    },
    data() {
        return {
            loading: true,
            items: null,
            message: null
        };
    },
    computed: {
        //

    },
    methods: {
        timedeltafordate,

        loadProfile() {
            this.loading = true
            axios
                .get(`${appConfig.API_URL}/api/online/users`)
                .then((response) => {
                    this.items = response.data.items;
                    this.message = response.data.message
                    this.loading = false
                })
                .catch((error) => {
                    this.message = error;
                    this.loading = false
                });
        },

    },

    created() {
        this.loadProfile()

        //
    },
};
</script>

<style >
</style>
