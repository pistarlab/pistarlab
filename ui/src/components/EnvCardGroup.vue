<template>
<div v-if="item.ident">
    <b-card class="h-100 card-shadow card-flyer" no-body header-bg-variant="white" header-text-variant="dark">

        <div class="text-center image-box">
            <b-card-img :src="`${appConfig.API_URL}/api/env_preview_image/env_${item.ident}`" alt="No Image Found" style="max-height:200px;"></b-card-img>
        </div>

        <b-card-body >
            <h4> {{ item.displayedName }}</h4>

        </b-card-body>

    </b-card>
</div>
</template>

<script>
import {
    appConfig
} from "../app.config";
export default {
    props: {
        item: Object
    },
    data() {
        return {
            appConfig

        };
    },
    mounted() {
        //
    },
    methods: {
        //

    },
    computed: {
        //

    },
    // Fetches posts when the component is created.
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style>

</style>
